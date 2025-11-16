// Function: sub_10D12F0
// Address: 0x10d12f0
//
__int64 __fastcall sub_10D12F0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5
    && !*(_QWORD *)(v5 + 8)
    && *(_BYTE *)v4 == 59
    && (v15 = *(_QWORD *)(v4 - 64)) != 0
    && (**a1 = v15, (v16 = *(_QWORD *)(v4 - 32)) != 0) )
  {
    *a1[1] = v16;
    v6 = *((_QWORD *)a3 - 4);
    v17 = *(_QWORD *)(v6 + 16);
    if ( !v17 || *(_QWORD *)(v17 + 8) )
      return 0;
    if ( *(_BYTE *)v6 == 58 )
    {
      v18 = *(_QWORD *)(v6 - 64);
      v19 = *(_QWORD *)(v6 - 32);
      v20 = *a1[2];
      if ( v18 == v20 && v19 )
      {
        *a1[3] = v19;
        return 1;
      }
      if ( v20 == v19 && v18 )
      {
        *a1[3] = v18;
        return 1;
      }
    }
  }
  else
  {
    v6 = *((_QWORD *)a3 - 4);
    v7 = *(_QWORD *)(v6 + 16);
    if ( !v7 || *(_QWORD *)(v7 + 8) )
      return 0;
  }
  if ( *(_BYTE *)v6 != 59 )
    return 0;
  v8 = *(_QWORD *)(v6 - 64);
  if ( !v8 )
    return 0;
  **a1 = v8;
  v9 = *(_QWORD *)(v6 - 32);
  if ( !v9 )
    return 0;
  *a1[1] = v9;
  v10 = *((_QWORD *)a3 - 8);
  v11 = *(_QWORD *)(v10 + 16);
  if ( !v11 || *(_QWORD *)(v11 + 8) || *(_BYTE *)v10 != 58 )
    return 0;
  v12 = *(_QWORD *)(v10 - 64);
  v13 = *(_QWORD *)(v10 - 32);
  v14 = *a1[2];
  if ( v12 != v14 || !v13 )
  {
    if ( v13 == v14 && v12 )
    {
      *a1[3] = v12;
      return 1;
    }
    return 0;
  }
  *a1[3] = v13;
  return 1;
}
