// Function: sub_1630860
// Address: 0x1630860
//
unsigned __int8 *__fastcall sub_1630860(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int8 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  double v15; // xmm4_8
  double v16; // xmm5_8
  unsigned __int8 *v17; // r13
  __int64 v18; // rdi

  v11 = (unsigned __int8 *)sub_162D4F0(a1, *(double *)a3.m128_u64, a4, a5, a6, a7, a8, a9, a10);
  v17 = v11;
  if ( (unsigned __int8 *)a1 == v11 )
  {
    sub_1623B10((__int64)v11, a2);
    return v17;
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 16);
    if ( (v18 & 4) != 0 )
    {
      a2 = (__int64)v11;
      sub_16302D0((const __m128i *)(v18 & 0xFFFFFFFFFFFFFFF8LL), v11, a3, a4, a5, a6, v15, v16, a9, a10);
    }
    sub_1623F10(a1, a2, v12, v13, v14);
    return v17;
  }
}
