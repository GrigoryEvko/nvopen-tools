// Function: sub_38809A0
// Address: 0x38809a0
//
__int64 ***__fastcall sub_38809A0(
        __int64 ***a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        _BYTE *a17,
        __int64 a18,
        __int8 *a19,
        size_t a20)
{
  __int64 v22; // rax
  __int64 v23; // r9
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 **v26; // r15
  __int64 v27; // r9
  double v28; // xmm4_8
  double v29; // xmm5_8

  v22 = sub_22077B0(0x2E0u);
  v26 = (__int64 **)v22;
  if ( !v22 )
  {
    if ( (unsigned __int8)sub_3880360(
                            0,
                            0,
                            a2,
                            a4,
                            a5,
                            v23,
                            a6,
                            a7,
                            a8,
                            a9,
                            v24,
                            v25,
                            a12,
                            a13,
                            a15,
                            a16,
                            (__int64)a17,
                            a18,
                            a19,
                            a20) )
    {
      *a1 = 0;
      return a1;
    }
    goto LABEL_6;
  }
  sub_1631D60(v22, a17, a18, a3);
  if ( !(unsigned __int8)sub_3880360(
                           v26,
                           0,
                           a2,
                           a4,
                           a5,
                           v27,
                           a6,
                           a7,
                           a8,
                           a9,
                           v28,
                           v29,
                           a12,
                           a13,
                           a15,
                           a16,
                           (__int64)a17,
                           a18,
                           a19,
                           a20) )
  {
LABEL_6:
    *a1 = v26;
    return a1;
  }
  *a1 = 0;
  sub_1633490(v26);
  j_j___libc_free_0((unsigned __int64)v26);
  return a1;
}
