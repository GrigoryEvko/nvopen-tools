// Function: sub_1D93A30
// Address: 0x1d93a30
//
_BOOL8 __fastcall sub_1D93A30(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, _DWORD *a5, __int64 a6, __int64 a7)
{
  char v8; // al
  _QWORD *v10; // r9
  _QWORD *v12; // r10
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // r14
  bool v16; // r11
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int8 v19; // r8
  _BOOL4 v20; // r8d
  __int64 v23; // [rsp+18h] [rbp-58h]
  unsigned __int8 v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v28; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v29[7]; // [rsp+38h] [rbp-38h] BYREF

  *a5 = 0;
  *a4 = 0;
  if ( (*(_BYTE *)a2 & 2) != 0 )
    return 0;
  if ( (*(_BYTE *)a2 & 1) != 0 )
    return 0;
  v8 = *(_BYTE *)a3;
  if ( (*(_BYTE *)a3 & 2) != 0 )
    return 0;
  if ( (v8 & 1) != 0 )
    return 0;
  if ( (*(_BYTE *)a2 & 0x10) == 0 )
    return 0;
  if ( (v8 & 0x10) == 0 )
    return 0;
  v10 = *(_QWORD **)(a2 + 16);
  if ( (unsigned int)((__int64)(v10[9] - v10[8]) >> 3) > 1 )
    return 0;
  v12 = *(_QWORD **)(a3 + 16);
  if ( (unsigned int)((__int64)(v12[9] - v12[8]) >> 3) > 1 || !*(_DWORD *)(a2 + 48) || !*(_DWORD *)(a3 + 48) )
    return 0;
  v13 = *(_QWORD *)(a2 + 24);
  v14 = *(_QWORD *)(a2 + 32);
  v23 = *(_QWORD *)(a3 + 24);
  v15 = *(_QWORD *)(a3 + 32);
  if ( !v13 && v10[1] != v10[7] + 320LL )
    v13 = v10[1];
  if ( !v14 && v10[1] != v10[7] + 320LL )
    v14 = v10[1];
  if ( v23 )
  {
LABEL_14:
    v16 = v14 == 0 || v13 == 0;
    if ( !v15 )
    {
      v15 = v12[1];
      if ( v15 == v12[7] + 320LL )
        return 0;
    }
    goto LABEL_15;
  }
  if ( v12[1] != v12[7] + 320LL )
  {
    v23 = v12[1];
    goto LABEL_14;
  }
  v16 = v13 == 0 || v14 == 0;
  if ( !v15 )
    return 0;
LABEL_15:
  if ( v16 )
    return 0;
  if ( v13 != v23 || v14 != v15 )
  {
    if ( v14 != v23 || v13 != v15 )
      return 0;
    goto LABEL_20;
  }
  if ( v14 == v23 && v13 == v15 )
  {
LABEL_20:
    if ( (v8 & 0x20) != 0 )
    {
      sub_1D920D0(a1, a3);
      v17 = *(_QWORD *)(a2 + 16);
      v26 = *(_QWORD *)(v17 + 32);
      v18 = *(_QWORD *)(a3 + 16);
      v27 = *(_QWORD *)(v18 + 32);
      v28 = (_QWORD *)(v17 + 24);
      v29[0] = v18 + 24;
      v19 = sub_1D931A0(a1, &v26, &v27, &v28, v29, a4, a5, v17, v18, 1);
      if ( v19 )
      {
        *(_QWORD *)(a6 + 16) = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(a7 + 16) = *(_QWORD *)(a3 + 16);
        v19 = sub_1D92B60(a1, &v26, &v27, (unsigned __int64)&v28, (unsigned __int64)v29, (char *)a6, (char *)a7);
        if ( v19 )
        {
          *(_DWORD *)(a6 + 4) = *(_DWORD *)(a2 + 4);
          *(_DWORD *)(a7 + 4) = *(_DWORD *)(a3 + 4);
        }
      }
      v25 = v19;
      sub_1D920D0(a1, a3);
      return (_BOOL4)v25;
    }
    return 0;
  }
  v26 = v10[4];
  v27 = v12[4];
  v28 = v10 + 3;
  v29[0] = v12 + 3;
  if ( !(unsigned __int8)sub_1D931A0(a1, &v26, &v27, &v28, v29, a4, a5, (__int64)v10, (__int64)v12, 1) )
    return 0;
  *(_QWORD *)(a6 + 16) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a7 + 16) = *(_QWORD *)(a3 + 16);
  v20 = sub_1D92B60(a1, &v26, &v27, (unsigned __int64)&v28, (unsigned __int64)v29, (char *)a6, (char *)a7);
  if ( v20 )
  {
    *(_DWORD *)(a6 + 4) = *(_DWORD *)(a2 + 4);
    *(_DWORD *)(a7 + 4) = *(_DWORD *)(a3 + 4);
  }
  return v20;
}
