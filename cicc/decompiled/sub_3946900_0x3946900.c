// Function: sub_3946900
// Address: 0x3946900
//
unsigned __int64 __fastcall sub_3946900(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  __int64 v6; // r9
  unsigned int v7; // r8d
  unsigned __int64 result; // rax
  __int64 v9; // rdi
  int v10; // r11d
  _QWORD *v11; // rcx
  int v12; // eax
  int v13; // edi
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  __int64 v17; // r8
  int v18; // r11d
  _QWORD *v19; // r10
  int v20; // eax
  __int64 v21; // r8
  _QWORD *v22; // r9
  unsigned int v23; // r12d
  int v24; // r10d
  __int64 v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-28h]
  __int64 v27; // [rsp+8h] [rbp-28h]

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v6 + 16LL * v7;
  v9 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
    return result;
  v10 = 1;
  v11 = 0;
  while ( v9 != -4 )
  {
    if ( v9 != -8 || v11 )
      result = (unsigned __int64)v11;
    v7 = (v5 - 1) & (v10 + v7);
    v9 = *(_QWORD *)(v6 + 16LL * v7);
    if ( a2 == v9 )
      return result;
    ++v10;
    v11 = (_QWORD *)result;
    result = v6 + 16LL * v7;
  }
  if ( !v11 )
    v11 = (_QWORD *)result;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_14:
    v26 = a3;
    sub_3946740(a1, 2 * v5);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      result = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = *(_DWORD *)(a1 + 16) + 1;
      a3 = v26;
      v11 = (_QWORD *)(v16 + 16 * result);
      v17 = *v11;
      if ( a2 != *v11 )
      {
        v18 = 1;
        v19 = 0;
        while ( v17 != -4 )
        {
          if ( v17 == -8 && !v19 )
            v19 = v11;
          result = v15 & (unsigned int)(v18 + result);
          v11 = (_QWORD *)(v16 + 16LL * (unsigned int)result);
          v17 = *v11;
          if ( a2 == *v11 )
            goto LABEL_10;
          ++v18;
        }
        if ( v19 )
          v11 = v19;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  result = v5 - *(_DWORD *)(a1 + 20) - v13;
  if ( (unsigned int)result <= v5 >> 3 )
  {
    v27 = a3;
    sub_3946740(a1, v5);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      result = (unsigned int)(v20 - 1);
      v21 = *(_QWORD *)(a1 + 8);
      v22 = 0;
      v23 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = 1;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      a3 = v27;
      v11 = (_QWORD *)(v21 + 16LL * v23);
      v25 = *v11;
      if ( a2 != *v11 )
      {
        while ( v25 != -4 )
        {
          if ( !v22 && v25 == -8 )
            v22 = v11;
          v23 = result & (v24 + v23);
          v11 = (_QWORD *)(v21 + 16LL * v23);
          v25 = *v11;
          if ( a2 == *v11 )
            goto LABEL_10;
          ++v24;
        }
        if ( v22 )
          v11 = v22;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v11 != -4 )
    --*(_DWORD *)(a1 + 20);
  *v11 = a2;
  v11[1] = a3;
  return result;
}
