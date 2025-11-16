// Function: sub_BD1850
// Address: 0xbd1850
//
__int64 __fastcall sub_BD1850(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r11d
  __int64 *v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 result; // rax
  int v12; // eax
  int v13; // edx
  unsigned __int8 **v14; // rbx
  __int64 v15; // rdx
  unsigned __int8 **i; // r13
  unsigned __int8 *v17; // rsi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r9d
  __int64 *v29; // r8
  unsigned int v30; // ebx
  __int64 v31; // rcx

  v2 = a1 + 32;
  v5 = *(_DWORD *)(a1 + 56);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_27;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 8LL * v9);
  result = *v10;
  if ( *v10 == a2 )
    return result;
  while ( result != -4096 )
  {
    if ( v8 || result != -8192 )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    result = *(_QWORD *)(v6 + 8LL * v9);
    if ( result == a2 )
      return result;
    ++v7;
    v8 = v10;
    v10 = (__int64 *)(v6 + 8LL * v9);
  }
  v12 = *(_DWORD *)(a1 + 48);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)(a1 + 32);
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_27:
    sub_BD1680(v2, 2 * v5);
    v18 = *(_DWORD *)(a1 + 56);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 40);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v20 + 8LL * v21);
      v22 = *v8;
      v13 = *(_DWORD *)(a1 + 48) + 1;
      if ( *v8 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -4096 )
        {
          if ( v22 == -8192 && !v24 )
            v24 = v8;
          v21 = v19 & (v23 + v21);
          v8 = (__int64 *)(v20 + 8LL * v21);
          v22 = *v8;
          if ( *v8 == a2 )
            goto LABEL_13;
          ++v23;
        }
        if ( v24 )
          v8 = v24;
      }
      goto LABEL_13;
    }
    goto LABEL_51;
  }
  if ( v5 - *(_DWORD *)(a1 + 52) - v13 <= v5 >> 3 )
  {
    sub_BD1680(v2, v5);
    v25 = *(_DWORD *)(a1 + 56);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 40);
      v28 = 1;
      v29 = 0;
      v30 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v27 + 8LL * v30);
      v31 = *v8;
      v13 = *(_DWORD *)(a1 + 48) + 1;
      if ( *v8 != a2 )
      {
        while ( v31 != -4096 )
        {
          if ( v31 == -8192 && !v29 )
            v29 = v8;
          v30 = v26 & (v28 + v30);
          v8 = (__int64 *)(v27 + 8LL * v30);
          v31 = *v8;
          if ( *v8 == a2 )
            goto LABEL_13;
          ++v28;
        }
        if ( v29 )
          v8 = v29;
      }
      goto LABEL_13;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 48) = v13;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v8 = a2;
  result = *(unsigned __int8 *)(a2 - 16);
  if ( (result & 2) != 0 )
  {
    v14 = *(unsigned __int8 ***)(a2 - 32);
    v15 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v15 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    v14 = (unsigned __int8 **)(a2 - result - 16);
  }
  for ( i = &v14[v15]; i != v14; ++v14 )
  {
    v17 = *v14;
    if ( *v14 )
    {
      result = *v17;
      if ( (unsigned __int8)(result - 5) > 0x1Fu )
      {
        if ( (_BYTE)result == 1 )
          result = sub_BD1B10(a1, *((_QWORD *)v17 + 17));
      }
      else
      {
        result = sub_BD1850(a1);
      }
    }
  }
  return result;
}
