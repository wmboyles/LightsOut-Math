# Lights Out Conjectures

1. If n is non-zero in A159257, then so is 2n+1. (Sutner also conjectured this ~1989)

   - **TRUE** I proved a more general version using the idea of tiling quiet patterns

2. The 4x4 board the the only n x n lights out board with rank deficiency n.

   - This might already be proven

3. The n x n boards with the greatest rank deficiency and index <= n are: n(rank deficiency)

   ```
   4(4), 9(8), 19(16), 39(32), ... , (5 * 2^k - 2)/2 (2^(k+1))
   ```

4. Let `d(n)` be the nullity of an n x n board. Then for all integers `n >= 0`, `d(n) >= d(n mod 30)`.

   - **PARTIALLY TRUE**.
     Can use tiling lemma: (`d(ij + i - 1) >= d(i)`) to show true for all remainders except `{11,16,19,23}`.
     - `d(n + 30) >= d(n)` is **false**.
       Counterexample: `d(94) = 4` and `d(64) = 28`.
     - `d(30k + r) >= d(r)` is true for r not in `{4,5,9,11,14,16,19,23,24,29}` b/c `d(r) = 0`.
     - Above **true** for `r+1` is a divisor of 30 b/c we know that `d((r+1)j + r) >= d(r)`.
       We can then substitute `30k/(r+1)` for `j`.
       This leaves `r` in `{11,16,19,23,24}` to deal with.
     - Above **false** for r = 11.
       `d(11) = 6` and `d(41) = 2`.
     - Above **false** for r = 16.
       `d(16) = 8` and `d(46) = 0`.
     - Above **false** for r = 19.
       `d(19) = 16` and `d(49) = 8`.
     - Above **false** for r = 23.
       `d(23) = 14` and `d(53) = 2`.
     - Above **true** for r = 24.
       We know that `d(5j + 4) >= d(4) = d(24)`.
       Substituting `6k+4` for `j` gives `d(30k + 24) >= d(24)`.
   - We can take away a general corollary of the tiling lemma from this: For integers k and 0 <= r < h, `d(hk + r) >= d(r)` if `r+1` is a divisor of `h`.

5. Number pattern follows the following pattern:
   `((s+1)*2^n -2)/2`
   where s is the starting number in the sequence `(4,5,14,16,17,24,30,32,34,41)`.

   Deficiency patterns are of one of two forms:

   1. `b*2^n`
   2. `b*2^n -2`

   <hr />
   Num(deficiency)

   - 4(4), 9(8), 19(16), 9(32), 79(64), 159(128), 319(256), 639(512)
     - Num Pattern: `(5*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 5(2), 11(6), 23(14), 47(30), 95(62), 191(126), 383(254), 767(510)
     - Num Pattern: `(6*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 14(4), 29(10), 59(22), 119(46), 239(94), 479(190), 959(382)
     - Num Pattern: `(15*2^n - 2)/2`
     - Def Pattern: `3*2^n - 2`
   - 16(8), 33(16), 67(32), 135(64), 271(128), 543(256)
     - Num Pattern: `(17*2^n - 2)/2`
     - Def Pattern: `4*2^n`
   - 17(2), 35(6), 71(14), 143(30), 287(62), 575(126)
     - Num Pattern: `(18*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 24(4), 49(8), 99(16), 199(32), 399(64), 799(128)
     - Num Pattern: `(25*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 30(20), 61(40), 123(80), 247(160), 495(320), 991(640)
     - Num Pattern: `(31*2^n - 2)/2`
     - Def Pattern: `10*2^n`
   - 32(20), 65(42), 131(86), 263(174), 527(350)
     - Num Pattern: `(33*2^n - 2)/2`
     - Def Pattern: `11*2^n - 2`
   - 34(4), 69(8), 139(16), 279(32), 559(64)
     - Num Pattern: `(35*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 41(2), 83(6), 167(14), 335(30), 671(62)
     - Num Pattern: `(42*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 44(4), 89(10), 179(22), 359(46), 719(94)
     - Num Pattern: `(45*2^n - 2)/2`
     - Def Pattern: `3*2^n - 2`
   - 50(8), 101(18), 203(38), 407(78), 815(158)
     - Num Pattern: `(51*2^n - 2)/2`
     - Def Pattern: `5*2^n - 2`
   - 53(2), 107(6), 215(14), 431(30), 862(62)
     - Num Pattern: `(54*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 54(4), 109(8), 219(16), 439(32), 879(64)
     - Num Pattern: `(55*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 62(24), 125(50), 251(102), 503(206), 1007(414)
     - Num Pattern: `(63*2^n - 2)/2`
     - Def Pattern: `13*2^n - 2`
   - 64(28), 129(56), 259(112), 519(224)
     - Num Pattern: `(65*2^n - 2)/2`
     - Def Pattern: `14*2^n`
   - 7\* 4(4), 149(10), 299(22), 599(46)
     - Num Pattern: `(75*2^n - 2)/2`
     - Def Pattern: `3*2^n - 2`
   - 84(12), 169(24), 339(48), 679(96)
     - Num Pattern: `(85*2^n - 2)/2`
     - Def Pattern: `6*2^n`
   - 92(20), 185(42), 371(86), 743(174)
     - Num Pattern: `(93*2^n - 2)/2`
     - Def Pattern: `11*2^n - 2`
   - 94(4), 189(10), 379(22), 759(46)
     - Num Pattern: `(95*2^n - 2)/2`
     - Def Pattern: `3*2^n - 2`
   - 98(20), 197(42), 395(86), 791(174)
     - Num Pattern: `(99*2^n - 2)/2`
     - Def Pattern: `11*2^n - 2`
   - 104(4), 209(10), 419(22), 839(46)
     - Num Pattern: `(105*2^n - 2)/2`
     - Def Pattern: `3*2^n - 2`
   - 113(2), 227(6), 455(14), 911(30)
     - Num Pattern: `(114*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 114(4), 229(8), 459(16), 919(32)
     - Num Pattern: `(115*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 118(8), 237(16), 475(32), 951(64)
     - Num Pattern: `(119*2^n - 2)/2`
     - Def Pattern: `4*2^n`
   - 124(4), 249(8), 499(16), 999(32)
     - Num Pattern: `(125*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 126(56), 253(112), 507(224)
     - Num Pattern: `(127*2^n - 2)/2`
     - Def Pattern: `28*2^n`
   - 128(56), 257(114), 515(230)
     - Num Pattern: `(129*2^n - 2)/2`
     - Def Pattern: `29*2^n - 2`
   - 134(4), 269(10), 539(22)
     - Num Pattern: `(135*2^n - 2)/2`
     - Def Pattern: `3*2^n - 2`
   - 137(2), 275(6), 551(14)
     - Num Pattern: `(138*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 144(4), 289(8), 579(16)
     - Num Pattern: `(145*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 152(8), 305(18), 611(38)
     - Num Pattern: `(153*2^n - 2)/2`
     - Def Pattern: `4*2^n`
   - 154(24), 309(48), 619(96)
     - Num Pattern: `(155*2^n - 2)/2`
     - Def Pattern: `12*2^n`
   - 155(6), 311(14), 623(30)
     - Num Pattern: `(156*2^n - 2)/2`
     - Def Pattern: `4*2^n - 2`
   - 161(2), 323(6), 647(14)
     - Num Pattern: `(162*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 164(24), 329(50), 659(102)
     - Num Pattern: `(165*2^n - 2)/2`
     - Def Pattern: `13*2^n - 2`
   - 170(36), 341(74), 683(150)
     - Num Pattern: `(171*2^n - 2)/2`
     - Def Pattern: `19*2^n - 2`
   - 173(2), 347(6), 695(14)
     - Num Pattern: `(174*2^n - 2)/2`
     - Def Pattern: `2*2^n - 2`
   - 174(4), 349(8), 699(16)
     - Num Pattern: `(175*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 184(4), 369(8), 739(16)
     - Num Pattern: `(185*2^n - 2)/2`
     - Def Pattern: `2*2^n`
   - 186(8), 373(16), 747(32)
     - Num Pattern: `(187*2^n - 2)/2`
     - Def Pattern: `4*2^n`
   - 188(24), 377(50), 755(102)
     - Num Pattern: `(189*2^n - 2)/2`
     - Def Pattern: `13*2^n - 2`
   - 194(28), 389(58), 779(118)
     - Num Pattern: `(195*2^n - 2)/2`
     - Def Pattern: `15*2^n - 2`
