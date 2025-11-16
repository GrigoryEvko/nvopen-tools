// Function: sub_8310F0
// Address: 0x8310f0
//
__int64 __fastcall sub_8310F0(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 *a4,
        int *a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        __m128i *a9)
{
  __int64 v9; // rax
  __int64 *v12; // r12
  char v14; // dl
  _BYTE *v15; // rdx
  __int64 *v16; // r15
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // edx
  _BOOL4 v21; // r15d
  __int64 i; // rax
  __int64 result; // rax
  __int64 v24; // [rsp+0h] [rbp-50h]
  int v25; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = a1;
  v12 = a4;
  v14 = *(_BYTE *)(a1 + 80);
  if ( v14 == 16 )
  {
    v9 = **(_QWORD **)(a1 + 88);
    v14 = *(_BYTE *)(v9 + 80);
  }
  if ( v14 == 24 )
    v9 = *(_QWORD *)(v9 + 88);
  v15 = a3;
  v16 = (__int64 *)(a3 + 68);
  v24 = v9;
  sub_82F430(a1, a2, v15, a4, 0, a8, 0, 0, a9, &v25, 0);
  v20 = *a5;
  if ( !v12 )
    v12 = v16;
  v21 = 0;
  if ( v20 )
    v21 = (*(_BYTE *)(a6 + 18) & 2) != 0;
  for ( i = *(_QWORD *)(*(_QWORD *)(v24 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)(i + 168);
  if ( *(_QWORD *)(result + 40) )
  {
    if ( v20 )
    {
      v26[0] = *(_QWORD *)(a6 + 68);
      sub_82FD20((const __m128i *)a6, v21, a1, a2, v25, a7 == 0, (__int64)v26);
    }
    else
    {
      if ( !(unsigned int)sub_830D50(a1, a2, v12, a9[1].m128i_i8[3] & 1, a6) )
        sub_6E6840((__int64)a9);
      *a5 = 1;
      v21 = 1;
    }
    return sub_82F1E0(a6, v21, (__int64)a9);
  }
  else if ( v20 )
  {
    result = sub_82F8F0((const __m128i *)a6, v21, a9, v17, v18, v19);
    *a5 = 0;
  }
  return result;
}
