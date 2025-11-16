// Function: sub_1647B40
// Address: 0x1647b40
//
__int64 __fastcall sub_1647B40(__int64 a1, __int64 a2)
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
  unsigned __int8 *v15; // rsi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // r8
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // r10d
  __int64 *v22; // r9
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 *v27; // r8
  unsigned int v28; // ebx
  __int64 v29; // rcx

  v2 = a1 + 32;
  v5 = *(_DWORD *)(a1 + 56);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 8LL * v9);
  result = *v10;
  if ( *v10 == a2 )
    return result;
  while ( result != -8 )
  {
    if ( v8 || result != -16 )
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
LABEL_24:
    sub_1647990(v2, 2 * v5);
    v16 = *(_DWORD *)(a1 + 56);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 40);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v18 + 8LL * v19);
      v20 = *v8;
      v13 = *(_DWORD *)(a1 + 48) + 1;
      if ( *v8 != a2 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v22 )
            v22 = v8;
          v19 = v17 & (v21 + v19);
          v8 = (__int64 *)(v18 + 8LL * v19);
          v20 = *v8;
          if ( *v8 == a2 )
            goto LABEL_13;
          ++v21;
        }
        if ( v22 )
          v8 = v22;
      }
      goto LABEL_13;
    }
    goto LABEL_48;
  }
  if ( v5 - *(_DWORD *)(a1 + 52) - v13 <= v5 >> 3 )
  {
    sub_1647990(v2, v5);
    v23 = *(_DWORD *)(a1 + 56);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 40);
      v26 = 1;
      v27 = 0;
      v28 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (__int64 *)(v25 + 8LL * v28);
      v29 = *v8;
      v13 = *(_DWORD *)(a1 + 48) + 1;
      if ( *v8 != a2 )
      {
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v27 )
            v27 = v8;
          v28 = v24 & (v26 + v28);
          v8 = (__int64 *)(v25 + 8LL * v28);
          v29 = *v8;
          if ( *v8 == a2 )
            goto LABEL_13;
          ++v26;
        }
        if ( v27 )
          v8 = v27;
      }
      goto LABEL_13;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_13:
  *(_DWORD *)(a1 + 48) = v13;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 52);
  *v8 = a2;
  result = 8LL * *(unsigned int *)(a2 + 8);
  v14 = (unsigned __int8 **)(a2 - result);
  if ( a2 - result != a2 )
  {
    do
    {
      v15 = *v14;
      if ( *v14 )
      {
        result = *v15;
        if ( (unsigned __int8)(result - 4) > 0x1Eu )
        {
          if ( (_BYTE)result == 1 )
            result = sub_1647DB0(a1, *((_QWORD *)v15 + 17));
        }
        else
        {
          result = sub_1647B40(a1);
        }
      }
      ++v14;
    }
    while ( (unsigned __int8 **)a2 != v14 );
  }
  return result;
}
