// Function: sub_1A1C950
// Address: 0x1a1c950
//
__int64 *__fastcall sub_1A1C950(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rsi
  char v6; // al
  char v8; // r8
  __int64 v10; // rcx
  char v11; // r8
  char v12; // r8
  char v13; // r9
  int v14; // eax
  int v15; // r8d
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // esi
  __int64 **v19; // rcx
  char v20; // r9
  __int64 **v21; // rax
  _QWORD *v22; // rax
  __int64 **v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 *v27; // [rsp+8h] [rbp-68h]
  __m128i v28; // [rsp+10h] [rbp-60h] BYREF
  __int16 v29; // [rsp+20h] [rbp-50h]
  __m128i v30; // [rsp+30h] [rbp-40h] BYREF
  __int16 v31; // [rsp+40h] [rbp-30h]

  v5 = *a3;
  if ( a4 == *a3 )
    return a3;
  v6 = *(_BYTE *)(v5 + 8);
  if ( v6 != 11 )
  {
    if ( v6 == 16 )
    {
      v10 = **(_QWORD **)(v5 + 16);
      v11 = *(_BYTE *)(v10 + 8);
      if ( v11 == 11 )
      {
        v8 = *(_BYTE *)(a4 + 8);
        goto LABEL_23;
      }
      if ( v11 != 15 )
      {
LABEL_9:
        v31 = 257;
        return sub_1A1C8D0(a2, 47, (__int64)a3, (__int64 **)a4, &v30);
      }
    }
    else
    {
      v10 = *a3;
      if ( v6 != 15 )
        goto LABEL_9;
    }
    v12 = *(_BYTE *)(a4 + 8);
    v13 = v12;
    if ( v12 == 16 )
      v13 = *(_BYTE *)(**(_QWORD **)(a4 + 16) + 8LL);
    if ( v13 != 11 )
      goto LABEL_15;
    if ( v6 == 16 )
    {
      if ( v12 == 16 )
        goto LABEL_32;
    }
    else if ( v12 != 16 )
    {
LABEL_32:
      v31 = 257;
      return sub_1A1C8D0(a2, 45, (__int64)a3, (__int64 **)a4, &v30);
    }
    v26 = (__int64)a3;
    v31 = 257;
    v29 = 257;
    v23 = (__int64 **)sub_15A9650(a1, v5);
    a3 = sub_1A1C8D0(a2, 45, v26, v23, &v28);
    return sub_1A1C8D0(a2, 47, (__int64)a3, (__int64 **)a4, &v30);
  }
  v8 = *(_BYTE *)(a4 + 8);
  if ( v8 == 11 )
  {
    if ( *(_DWORD *)(a4 + 8) >> 8 > *(_DWORD *)(v5 + 8) >> 8 )
    {
      v31 = 257;
      return sub_1A1C8D0(a2, 37, (__int64)a3, (__int64 **)a4, &v30);
    }
    goto LABEL_9;
  }
  v10 = *a3;
LABEL_23:
  if ( v8 == 16 )
    v20 = *(_BYTE *)(**(_QWORD **)(a4 + 16) + 8LL);
  else
    v20 = v8;
  if ( v20 == 15 )
  {
    if ( v6 == 16 )
    {
      if ( v8 == 16 )
        goto LABEL_28;
    }
    else if ( v8 != 16 )
    {
LABEL_28:
      v31 = 257;
      return sub_1A1C8D0(a2, 46, (__int64)a3, (__int64 **)a4, &v30);
    }
    v27 = a3;
    v31 = 257;
    v29 = 257;
    v24 = sub_15A9650(a1, a4);
    v17 = (__int64)v27;
    v18 = 47;
    v19 = (__int64 **)v24;
LABEL_21:
    a3 = sub_1A1C8D0(a2, v18, v17, v19, &v28);
    return sub_1A1C8D0(a2, 46, (__int64)a3, (__int64 **)a4, &v30);
  }
LABEL_15:
  if ( v6 != 15 )
    goto LABEL_9;
  if ( *(_BYTE *)(a4 + 8) != 15 )
    goto LABEL_9;
  v14 = *(_DWORD *)(v10 + 8) >> 8;
  v15 = *(_DWORD *)(a4 + 8) >> 8;
  if ( v15 == v14 )
    goto LABEL_9;
  v25 = (__int64)a3;
  if ( v15 && v14 )
  {
    v31 = 257;
    v29 = 257;
    v16 = sub_15A9650(a1, v5);
    v17 = v25;
    v18 = 45;
    v19 = (__int64 **)v16;
    goto LABEL_21;
  }
  v21 = (__int64 **)sub_1646BA0(*(__int64 **)(v5 + 24), v15);
  v31 = 257;
  v22 = sub_1A1C8D0(a2, 48, v25, v21, &v30);
  v31 = 257;
  return sub_1A1C8D0(a2, 47, (__int64)v22, (__int64 **)a4, &v30);
}
