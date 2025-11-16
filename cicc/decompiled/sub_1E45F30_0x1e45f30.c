// Function: sub_1E45F30
// Address: 0x1e45f30
//
__int64 __fastcall sub_1E45F30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // r9
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  int v13; // r15d
  int v14; // eax
  int v15; // edi
  __int64 v16; // rsi
  unsigned int v17; // r8d
  unsigned int i; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 result; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // r8
  __int64 v26; // rdi
  __int64 v27; // rdx
  int v28; // r8d
  int v29; // eax
  unsigned int v30; // edx
  int v31; // [rsp+4h] [rbp-3Ch]
  int v32; // [rsp+8h] [rbp-38h]
  int v33; // [rsp+Ch] [rbp-34h]

  v6 = a1 + 40;
  v7 = sub_1E45EB0(a2, a3);
  v8 = (_QWORD *)(a1 + 40);
  v9 = v7;
  v10 = *(_QWORD **)(a1 + 48);
  if ( v10 )
  {
    do
    {
      while ( 1 )
      {
        v11 = v10[2];
        v12 = v10[3];
        if ( v10[4] >= v9 )
          break;
        v10 = (_QWORD *)v10[3];
        if ( !v12 )
          goto LABEL_6;
      }
      v8 = v10;
      v10 = (_QWORD *)v10[2];
    }
    while ( v11 );
LABEL_6:
    if ( (_QWORD *)v6 != v8 && v8[4] > v9 )
      v8 = (_QWORD *)(a1 + 40);
  }
  v13 = *(_DWORD *)(a1 + 128);
  v32 = *((_DWORD *)v8 + 10);
  v31 = *(_DWORD *)(a1 + 136);
  v14 = sub_1E404B0(a1, v9);
  v15 = *(_DWORD *)(a3 + 40);
  v33 = v14;
  if ( v15 == 1 )
  {
    v17 = 0;
  }
  else
  {
    v16 = *(_QWORD *)(a3 + 32);
    v17 = 0;
    for ( i = 1; i != v15; i += 2 )
    {
      while ( *(_QWORD *)(a3 + 24) != *(_QWORD *)(v16 + 40LL * (i + 1) + 24) )
      {
        i += 2;
        if ( v15 == i )
          goto LABEL_14;
      }
      v19 = i;
      v17 = *(_DWORD *)(v16 + 40 * v19 + 8);
    }
  }
LABEL_14:
  v20 = sub_1E69D00(*(_QWORD *)(a1 + 152), v17);
  v21 = sub_1E45EB0(a2, v20);
  v22 = v21;
  if ( !v21 )
    return 1;
  result = *(_QWORD *)(*(_QWORD *)(v21 + 8) + 16LL);
  LOBYTE(result) = *(_WORD *)result == 45 || *(_WORD *)result == 0;
  if ( !(_BYTE)result )
  {
    v24 = *(_QWORD **)(a1 + 48);
    if ( v24 )
    {
      v25 = (_QWORD *)(a1 + 40);
      do
      {
        while ( 1 )
        {
          v26 = v24[2];
          v27 = v24[3];
          if ( v24[4] >= v22 )
            break;
          v24 = (_QWORD *)v24[3];
          if ( !v27 )
            goto LABEL_21;
        }
        v25 = v24;
        v24 = (_QWORD *)v24[2];
      }
      while ( v26 );
LABEL_21:
      if ( (_QWORD *)v6 != v25 && v25[4] <= v22 )
        v6 = (__int64)v25;
    }
    v28 = sub_1E404B0(a1, v22);
    v29 = (*(_DWORD *)(v6 + 40) - *(_DWORD *)(a1 + 128)) / *(_DWORD *)(a1 + 136);
    v30 = (*(_DWORD *)(v6 + 40) - *(_DWORD *)(a1 + 128)) % *(_DWORD *)(a1 + 136);
    LOBYTE(v29) = (v32 - v13) % v31 < v30;
    LOBYTE(v30) = v33 >= v28;
    return v30 | v29;
  }
  return result;
}
