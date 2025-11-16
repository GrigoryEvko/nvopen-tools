// Function: sub_33E2690
// Address: 0x33e2690
//
__int16 __fastcall sub_33E2690(__int64 a1, __int64 a2, unsigned int a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r14
  _DWORD *v9; // r13
  __int64 v10; // rbx
  __int16 v11; // ax
  __int64 v12; // rdx
  char v13; // bl
  int v14; // eax
  unsigned int v15; // ebx
  __int16 result; // ax
  unsigned __int16 v17; // cx
  unsigned int v18; // ebx
  int v19; // eax
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int16 v23; // [rsp+0h] [rbp-40h] BYREF
  __int64 v24; // [rsp+8h] [rbp-38h]

  v7 = sub_33DFBC0(a2, a3, 0, a4, a5, a6);
  if ( !v7 )
    return 0;
  v8 = *(_QWORD *)(v7 + 96);
  v9 = *(_DWORD **)(a1 + 16);
  v10 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v23 = v11;
  v24 = v12;
  if ( v11 )
  {
    v17 = v11 - 17;
    if ( (unsigned __int16)(v11 - 10) <= 6u || (unsigned __int16)(v11 - 126) <= 0x31u )
    {
      if ( v17 > 0xD3u )
        goto LABEL_29;
    }
    else if ( v17 > 0xD3u )
    {
      goto LABEL_5;
    }
  }
  else
  {
    v13 = sub_3007030((__int64)&v23);
    if ( !sub_30070B0((__int64)&v23) )
    {
      if ( !v13 )
      {
LABEL_5:
        v14 = v9[15];
        goto LABEL_6;
      }
LABEL_29:
      v14 = v9[16];
LABEL_6:
      if ( v14 != 1 )
        goto LABEL_7;
      goto LABEL_17;
    }
  }
  v14 = v9[17];
  if ( v14 == 1 )
  {
LABEL_17:
    v18 = *(_DWORD *)(v8 + 32);
    if ( v18 > 0x40 )
    {
      v19 = sub_C444A0(v8 + 24);
      if ( v19 != v18 - 1 )
      {
        if ( v18 == v19 )
          return 256;
        return 0;
      }
      return 257;
    }
    v20 = *(_QWORD *)(v8 + 24);
    if ( v20 == 1 )
      return 257;
    if ( !v20 )
      return 256;
    return 0;
  }
LABEL_7:
  if ( v14 == 2 )
  {
    v15 = *(_DWORD *)(v8 + 32);
    if ( v15 )
    {
      if ( v15 > 0x40 )
      {
        if ( v15 != (unsigned int)sub_C445E0(v8 + 24) )
        {
          if ( v15 == (unsigned int)sub_C444A0(v8 + 24) )
            return 256;
          return 0;
        }
        return 257;
      }
      if ( *(_QWORD *)(v8 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) )
      {
        if ( !*(_QWORD *)(v8 + 24) )
          return 256;
        return 0;
      }
    }
    return 257;
  }
  if ( v14 )
    BUG();
  v21 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v22 = *v21 & 1LL;
  else
    LOBYTE(v22) = (unsigned __int8)v21 & 1;
  LOBYTE(result) = v22;
  HIBYTE(result) = 1;
  return result;
}
