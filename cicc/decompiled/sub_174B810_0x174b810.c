// Function: sub_174B810
// Address: 0x174b810
//
__int64 __fastcall sub_174B810(
        __int64 *a1,
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
  __int64 v10; // rax
  unsigned int v11; // r14d
  int v12; // ebx
  double v13; // xmm4_8
  double v14; // xmm5_8
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 v18; // r15
  __int64 *v19; // r13
  unsigned int v20; // ebx
  unsigned int v21; // eax
  __int64 v22; // r14
  _QWORD *v23; // rax
  _QWORD *v24; // r12
  __int64 v26[2]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v27; // [rsp+10h] [rbp-40h]

  v10 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v10 = **(_QWORD **)(v10 + 16);
  v11 = *(_DWORD *)(v10 + 8) >> 8;
  v12 = sub_16431D0(**(_QWORD **)(a2 - 24));
  if ( v12 == 8 * (unsigned int)sub_15A9520(a1[333], v11) )
    return sub_174B490(a1, a2, a3, a4, a5, a6, v13, v14, a9, a10);
  v15 = a1[333];
  v16 = sub_16498A0(a2);
  v17 = (__int64 *)sub_15A9620(v15, v16, v11);
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    v17 = sub_16463B0(v17, *(_QWORD *)(*(_QWORD *)a2 + 32LL));
  v18 = a1[1];
  v19 = *(__int64 **)(a2 - 24);
  v27 = 257;
  v20 = sub_16431D0(*v19);
  v21 = sub_16431D0((__int64)v17);
  if ( v20 < v21 )
  {
    v19 = (__int64 *)sub_1708970(v18, 37, (__int64)v19, (__int64 **)v17, v26);
  }
  else if ( v20 > v21 )
  {
    v19 = (__int64 *)sub_1708970(v18, 36, (__int64)v19, (__int64 **)v17, v26);
  }
  v22 = *(_QWORD *)a2;
  v27 = 257;
  v23 = sub_1648A60(56, 1u);
  v24 = v23;
  if ( v23 )
    sub_15FD410((__int64)v23, (__int64)v19, v22, (__int64)v26, 0);
  return (__int64)v24;
}
