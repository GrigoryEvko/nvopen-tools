// Function: sub_324C1A0
// Address: 0x324c1a0
//
__int64 __fastcall sub_324C1A0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 *v6; // rdx
  __int64 result; // rax
  __int64 *v8; // r8
  int v9; // r9d
  __int64 *v10; // rcx
  __int64 v11; // r10
  int v12; // eax
  int v13; // edx
  __int64 v14; // rsi
  __int64 *v15; // rdi
  int v16; // r10d
  __int64 *v17; // r8
  unsigned int v18; // r9d
  __int64 v19; // rsi
  int v20; // r10d
  unsigned int v21; // r9d

  v4 = a1 + 232;
  v5 = *(_DWORD *)(a1 + 256);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 232);
    goto LABEL_14;
  }
  v6 = *(__int64 **)(a1 + 240);
  result = *v6;
  if ( !*v6 )
    return result;
  v8 = *(__int64 **)(a1 + 240);
  v9 = 1;
  v10 = 0;
  LODWORD(v11) = 0;
  while ( result != -4096 )
  {
    if ( v10 || result != -8192 )
      v8 = v10;
    v11 = (v5 - 1) & ((_DWORD)v11 + v9);
    result = v6[2 * v11];
    if ( !result )
      return result;
    ++v9;
    v10 = v8;
    v8 = &v6[2 * v11];
  }
  v12 = *(_DWORD *)(a1 + 248);
  if ( !v10 )
    v10 = v8;
  ++*(_QWORD *)(a1 + 232);
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
  {
LABEL_14:
    sub_324BD10(v4, 2 * v5);
    result = *(unsigned int *)(a1 + 256);
    if ( (_DWORD)result )
    {
      v10 = *(__int64 **)(a1 + 240);
      v14 = *v10;
      v13 = *(_DWORD *)(a1 + 248) + 1;
      if ( !*v10 )
        goto LABEL_10;
      result = (unsigned int)(result - 1);
      v15 = *(__int64 **)(a1 + 240);
      v16 = 1;
      v17 = 0;
      v18 = 0;
      while ( v14 != -4096 )
      {
        if ( v14 == -8192 && !v17 )
          v17 = v15;
        v18 = result & (v16 + v18);
        v15 = &v10[2 * v18];
        v14 = *v15;
        if ( !*v15 )
        {
LABEL_29:
          v10 = v15;
          goto LABEL_10;
        }
        ++v16;
      }
LABEL_18:
      v10 = v15;
      if ( v17 )
        v10 = v17;
      goto LABEL_10;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 248);
    BUG();
  }
  result = v5 - *(_DWORD *)(a1 + 252) - v13;
  if ( (unsigned int)result <= v5 >> 3 )
  {
    sub_324BD10(v4, v5);
    result = *(unsigned int *)(a1 + 256);
    if ( (_DWORD)result )
    {
      v10 = *(__int64 **)(a1 + 240);
      v19 = *v10;
      v13 = *(_DWORD *)(a1 + 248) + 1;
      if ( !*v10 )
        goto LABEL_10;
      result = (unsigned int)(result - 1);
      v15 = *(__int64 **)(a1 + 240);
      v20 = 1;
      v17 = 0;
      v21 = 0;
      while ( v19 != -4096 )
      {
        if ( !v17 && v19 == -8192 )
          v17 = v15;
        v21 = result & (v20 + v21);
        v15 = &v10[2 * v21];
        v19 = *v15;
        if ( !*v15 )
          goto LABEL_29;
        ++v20;
      }
      goto LABEL_18;
    }
    goto LABEL_41;
  }
LABEL_10:
  *(_DWORD *)(a1 + 248) = v13;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 252);
  *v10 = 0;
  v10[1] = a2;
  return result;
}
