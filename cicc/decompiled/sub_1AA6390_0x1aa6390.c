// Function: sub_1AA6390
// Address: 0x1aa6390
//
__int64 __fastcall sub_1AA6390(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD *v12; // rbx
  _QWORD *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r15
  unsigned __int64 *v16; // rcx
  unsigned __int64 v17; // rdx
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 result; // rax

  v12 = *(_QWORD **)a2;
  if ( !*(_QWORD *)a2 )
  {
    sub_164D160(0, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    BUG();
  }
  sub_164D160((__int64)(v12 - 3), a3, a4, a5, a6, a7, a8, a9, a10, a11);
  if ( (*((_BYTE *)v12 - 1) & 0x20) != 0 && (*(_BYTE *)(a3 + 23) & 0x20) == 0 )
    sub_164B7C0(a3, (__int64)(v12 - 3));
  v13 = *(_QWORD **)a2;
  v15 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
  v14 = *(_QWORD *)a2 - 24LL;
  sub_157EA20(a1, v14);
  v16 = (unsigned __int64 *)v13[1];
  v17 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
  *v16 = v17 | *v16 & 7;
  *(_QWORD *)(v17 + 8) = v16;
  *v13 &= 7uLL;
  v13[1] = 0;
  result = sub_164BEC0(v14, v14, v17, (__int64)v16, a4, a5, a6, a7, v18, v19, a10, a11);
  *(_QWORD *)a2 = v15;
  return result;
}
