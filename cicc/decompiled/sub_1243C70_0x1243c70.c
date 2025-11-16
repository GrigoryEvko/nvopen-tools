// Function: sub_1243C70
// Address: 0x1243c70
//
unsigned __int64 __fastcall sub_1243C70(__int64 a1, int a2, __int64 a3)
{
  unsigned int v5; // esi
  __int64 v6; // r9
  unsigned int v7; // r8d
  unsigned __int64 result; // rax
  int v9; // edi
  int v10; // r11d
  _DWORD *v11; // rcx
  int v12; // eax
  int v13; // edi
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  int v17; // r8d
  int v18; // r11d
  _DWORD *v19; // r10
  int v20; // eax
  __int64 v21; // r8
  _DWORD *v22; // r9
  unsigned int v23; // r13d
  int v24; // r10d
  int v25; // esi
  __int64 v26; // [rsp+8h] [rbp-28h]
  __int64 v27; // [rsp+8h] [rbp-28h]

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (v5 - 1) & (37 * a2);
  result = v6 + 16LL * v7;
  v9 = *(_DWORD *)result;
  if ( a2 == *(_DWORD *)result )
    goto LABEL_3;
  v10 = 1;
  v11 = 0;
  while ( v9 != -1 )
  {
    if ( v9 != -2 || v11 )
      result = (unsigned __int64)v11;
    v7 = (v5 - 1) & (v10 + v7);
    v9 = *(_DWORD *)(v6 + 16LL * v7);
    if ( a2 == v9 )
      goto LABEL_3;
    ++v10;
    v11 = (_DWORD *)result;
    result = v6 + 16LL * v7;
  }
  if ( !v11 )
    v11 = (_DWORD *)result;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_14:
    v26 = a3;
    sub_1243A90(a1, 2 * v5);
    v14 = *(_DWORD *)(a1 + 24);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = *(_QWORD *)(a1 + 8);
      result = (v14 - 1) & (unsigned int)(37 * a2);
      v13 = *(_DWORD *)(a1 + 16) + 1;
      a3 = v26;
      v11 = (_DWORD *)(v16 + 16 * result);
      v17 = *v11;
      if ( a2 != *v11 )
      {
        v18 = 1;
        v19 = 0;
        while ( v17 != -1 )
        {
          if ( v17 == -2 && !v19 )
            v19 = v11;
          result = v15 & (unsigned int)(v18 + result);
          v11 = (_DWORD *)(v16 + 16LL * (unsigned int)result);
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
    sub_1243A90(a1, v5);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      result = (unsigned int)(v20 - 1);
      v21 = *(_QWORD *)(a1 + 8);
      v22 = 0;
      v23 = result & (37 * a2);
      v24 = 1;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      a3 = v27;
      v11 = (_DWORD *)(v21 + 16LL * v23);
      v25 = *v11;
      if ( a2 != *v11 )
      {
        while ( v25 != -1 )
        {
          if ( !v22 && v25 == -2 )
            v22 = v11;
          v23 = result & (v24 + v23);
          v11 = (_DWORD *)(v21 + 16LL * v23);
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
  if ( *v11 != -1 )
    --*(_DWORD *)(a1 + 20);
  *v11 = a2;
  *((_QWORD *)v11 + 1) = a3;
LABEL_3:
  *(_DWORD *)(a1 + 32) = a2 + 1;
  return result;
}
