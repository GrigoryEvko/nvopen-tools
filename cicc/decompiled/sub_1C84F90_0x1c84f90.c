// Function: sub_1C84F90
// Address: 0x1c84f90
//
void __fastcall sub_1C84F90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        unsigned __int8 a15,
        _QWORD *a16,
        __int64 a17)
{
  int v17; // r15d
  unsigned int v18; // ebx
  __int64 v20; // rax
  __int64 v21; // r14
  unsigned int v22; // esi
  unsigned int v23; // esi
  __int64 *v24; // r14
  __int64 **v25; // r15
  __int64 **v26; // r8
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // rdx
  int v30; // [rsp+4h] [rbp-4Ch]

  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
    BUG();
  v17 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 15 )
    BUG();
  v18 = 1;
  if ( a5 )
    v18 = a5;
  v30 = *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8;
  v20 = v18 | 4;
  v21 = v20 & -v20;
  v22 = v20 & -(v18 | 4);
  if ( !(unsigned __int8)sub_1C2F070(a17) && byte_4FBD8E0 && v17 == 101 )
    v17 = 5;
  if ( v21 == 1 )
  {
    v23 = v18 / v22;
    v24 = (__int64 *)sub_1643330(a16);
  }
  else
  {
    v23 = v18 / v22;
    if ( v21 == 2 )
      v24 = (__int64 *)sub_1643340(a16);
    else
      v24 = (__int64 *)sub_1643350(a16);
  }
  if ( v23 != 1 )
  {
    if ( v23 == 2 )
      v24 = sub_1645D80(v24, 2);
    else
      v24 = sub_16463B0(v24, v23);
  }
  v25 = (__int64 **)sub_1646BA0(v24, v17);
  v26 = (__int64 **)sub_1646BA0(v24, v30);
  if ( *(_BYTE *)(a1 + 16) == 78
    && (v29 = *(_QWORD *)(a1 - 24), !*(_BYTE *)(v29 + 16))
    && (*(_BYTE *)(v29 + 33) & 0x20) != 0
    && *(_DWORD *)(v29 + 36) == 135 )
  {
    sub_1C82A50(a1, a2, v25, a3, v26, a4, a7, a8, a9, a10, v27, v28, a13, a14, v18, a6, a15, (__int64)a16, a17);
  }
  else
  {
    sub_1C81350(
      (__int64 *)a1,
      (__int64)v24,
      a2,
      v25,
      a3,
      v26,
      a7,
      a8,
      a9,
      a10,
      v27,
      v28,
      a13,
      a14,
      a4,
      v18,
      a6,
      a15,
      (__int64)a16,
      a17);
  }
}
