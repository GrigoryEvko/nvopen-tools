// Function: sub_BCDE10
// Address: 0xbcde10
//
__int64 __fastcall sub_BCDE10(__int64 *a1, int a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r14d
  __int64 v8; // r8
  __int64 v9; // rdx
  unsigned int j; // r10d
  __int64 v11; // rcx
  __int64 *v12; // r9
  unsigned int v13; // r10d
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // rsi
  int v17; // r9d
  __int64 v18; // r8
  unsigned int i; // eax
  __int64 *v20; // rdi
  unsigned int v21; // eax
  __int64 result; // rax
  __int64 *v23; // r14
  int v24; // r8d
  int v25; // ecx
  int v26; // ecx
  int v27; // ecx
  int v28; // r9d
  __int64 v29; // rdi
  unsigned int k; // eax
  __int64 *v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)*a1;
  v5 = *(_DWORD *)(v4 + 3088);
  v6 = v4 + 3064;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 3064);
LABEL_9:
    sub_BCD760(v6, 2 * v5);
    v14 = *(_DWORD *)(v4 + 3088);
    v9 = 0;
    if ( v14 )
    {
      v15 = v14 - 1;
      v17 = 1;
      v18 = 0;
      for ( i = v15
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)(37 * a2 - 1)
                  | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
               ^ (484763065 * (37 * a2 - 1))); ; i = v15 & v21 )
      {
        v16 = *(_QWORD *)(v4 + 3072);
        v9 = v16 + 24LL * i;
        v20 = *(__int64 **)v9;
        if ( a1 == *(__int64 **)v9 && a2 == *(_DWORD *)(v9 + 8) && *(_BYTE *)(v9 + 12) )
          break;
        if ( v20 == (__int64 *)-4096LL )
        {
          if ( *(_DWORD *)(v9 + 8) == -1 && *(_BYTE *)(v9 + 12) )
          {
LABEL_50:
            if ( v18 )
              v9 = v18;
            goto LABEL_22;
          }
        }
        else if ( v20 == (__int64 *)-8192LL && *(_DWORD *)(v9 + 8) == -2 && *(_BYTE *)(v9 + 12) != 1 && !v18 )
        {
          v18 = v16 + 24LL * i;
        }
        v21 = v17 + i;
        ++v17;
      }
    }
    goto LABEL_22;
  }
  v7 = 1;
  v8 = *(_QWORD *)(v4 + 3072);
  v9 = 0;
  for ( j = (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * a2 - 1)
              | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
           ^ (484763065 * (37 * a2 - 1)))
          & (v5 - 1); ; j = (v5 - 1) & v13 )
  {
    v11 = v8 + 24LL * j;
    v12 = *(__int64 **)v11;
    if ( a1 == *(__int64 **)v11 && a2 == *(_DWORD *)(v11 + 8) && *(_BYTE *)(v11 + 12) )
    {
      result = *(_QWORD *)(v11 + 16);
      v23 = (__int64 *)(v11 + 16);
      if ( !result )
        goto LABEL_26;
      return result;
    }
    if ( v12 == (__int64 *)-4096LL )
      break;
    if ( v12 == (__int64 *)-8192LL && *(_DWORD *)(v11 + 8) == -2 && *(_BYTE *)(v11 + 12) != 1 && !v9 )
      v9 = v8 + 24LL * j;
LABEL_7:
    v13 = v7 + j;
    ++v7;
  }
  if ( *(_DWORD *)(v11 + 8) != -1 || !*(_BYTE *)(v11 + 12) )
    goto LABEL_7;
  if ( !v9 )
    v9 = v8 + 24LL * j;
  v25 = *(_DWORD *)(v4 + 3080);
  ++*(_QWORD *)(v4 + 3064);
  v24 = v25 + 1;
  if ( 4 * (v25 + 1) >= 3 * v5 )
    goto LABEL_9;
  if ( v5 - *(_DWORD *)(v4 + 3084) - v24 > v5 >> 3 )
    goto LABEL_23;
  sub_BCD760(v6, v5);
  v26 = *(_DWORD *)(v4 + 3088);
  v9 = 0;
  if ( v26 )
  {
    v27 = v26 - 1;
    v28 = 1;
    v18 = 0;
    for ( k = v27
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(37 * a2 - 1)
                | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))) >> 31)
             ^ (484763065 * (37 * a2 - 1))); ; k = v27 & v32 )
    {
      v29 = *(_QWORD *)(v4 + 3072);
      v9 = v29 + 24LL * k;
      v31 = *(__int64 **)v9;
      if ( a1 == *(__int64 **)v9 && a2 == *(_DWORD *)(v9 + 8) && *(_BYTE *)(v9 + 12) )
        break;
      if ( v31 == (__int64 *)-4096LL )
      {
        if ( *(_DWORD *)(v9 + 8) == -1 && *(_BYTE *)(v9 + 12) )
          goto LABEL_50;
      }
      else if ( v31 == (__int64 *)-8192LL && *(_DWORD *)(v9 + 8) == -2 && *(_BYTE *)(v9 + 12) != 1 && !v18 )
      {
        v18 = v29 + 24LL * k;
      }
      v32 = v28 + k;
      ++v28;
    }
  }
LABEL_22:
  v24 = *(_DWORD *)(v4 + 3080) + 1;
LABEL_23:
  *(_DWORD *)(v4 + 3080) = v24;
  if ( *(_QWORD *)v9 != -4096 || *(_DWORD *)(v9 + 8) != -1 || !*(_BYTE *)(v9 + 12) )
    --*(_DWORD *)(v4 + 3084);
  *(_QWORD *)v9 = a1;
  v23 = (__int64 *)(v9 + 16);
  *(_DWORD *)(v9 + 8) = a2;
  *(_BYTE *)(v9 + 12) = 1;
  *(_QWORD *)(v9 + 16) = 0;
LABEL_26:
  result = sub_A777F0(0x28u, (__int64 *)(v4 + 2640));
  if ( result )
  {
    v33 = result;
    sub_BCBC80(result, a1, a2, 0x12u);
    result = v33;
  }
  *v23 = result;
  return result;
}
