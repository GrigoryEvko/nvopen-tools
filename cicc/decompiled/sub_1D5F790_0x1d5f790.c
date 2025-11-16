// Function: sub_1D5F790
// Address: 0x1d5f790
//
__int64 __fastcall sub_1D5F790(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        int *a12,
        __int64 a13,
        __int64 a14,
        _QWORD *a15)
{
  __int64 *v17; // rax
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 v20; // r12
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rdx
  _QWORD **v24; // rax
  _QWORD *v25; // r13
  _QWORD *v26; // rsi
  __int64 *v28; // rax
  double v29; // xmm4_8
  double v30; // xmm5_8
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v33; // rsi
  char v34; // [rsp+7h] [rbp-49h]
  _QWORD *v36; // [rsp+18h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v17 = *(__int64 **)(a1 - 8);
    v18 = *v17;
    if ( *(_BYTE *)(*v17 + 16) != 61 )
      goto LABEL_3;
  }
  else
  {
    v18 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v18 + 16) != 61 )
    {
LABEL_3:
      if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
        v19 = *(__int64 **)(v18 - 8);
      else
        v19 = (__int64 *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
      v20 = a1;
      sub_1D5BD90(a2, a1, 0, *v19);
      *a12 = 0;
      v34 = 0;
      if ( *(_QWORD *)(v18 + 8) )
        goto LABEL_6;
LABEL_18:
      sub_1D5C680(a2, v18, 0, a3, a4, a5, a6, v21, v22, a9, a10);
      goto LABEL_6;
    }
  }
  v34 = sub_1D5EF60(a15, (_QWORD *)v18) ^ 1;
  if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
    v28 = *(__int64 **)(v18 - 8);
  else
    v28 = (__int64 *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
  v20 = sub_1D5BAE0(a2, (unsigned __int8 *)a1, *v28, *(__int64 ***)a1);
  sub_1D5B980(a2, a1, v20, a3, a4, a5, a6, v29, v30, a9, a10);
  sub_1D5C680(a2, a1, 0, a3, a4, a5, a6, v31, v32, a9, a10);
  *a12 = 0;
  if ( !*(_QWORD *)(v18 + 8) )
    goto LABEL_18;
LABEL_6:
  if ( *(_BYTE *)(v20 + 16) <= 0x17u )
    return v20;
  v36 = (_QWORD *)v20;
  v23 = *(_QWORD *)v20;
  if ( (*(_BYTE *)(v20 + 23) & 0x40) == 0 )
  {
    v25 = *(_QWORD **)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
    if ( *v25 != v23 )
      goto LABEL_9;
LABEL_20:
    v33 = v20;
    v20 = (__int64)v25;
    sub_1D5C680(a2, v33, (__int64)v25, a3, a4, a5, a6, v21, v22, a9, a10);
    return v20;
  }
  v24 = *(_QWORD ***)(v20 - 8);
  v25 = *v24;
  if ( **v24 == v23 )
    goto LABEL_20;
LABEL_9:
  v26 = (_QWORD *)v20;
  if ( a13 )
  {
    sub_14EF3D0(a13, &v36);
    v26 = v36;
  }
  *a12 = (unsigned __int8)(sub_1D5EF60(a15, v26) | v34) ^ 1;
  return v20;
}
