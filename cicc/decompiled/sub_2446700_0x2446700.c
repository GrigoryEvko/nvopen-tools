// Function: sub_2446700
// Address: 0x2446700
//
__int64 *__fastcall sub_2446700(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  __int64 *v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v17; // rdx
  __int64 v18; // r10
  int v19; // edi
  unsigned int v20; // esi
  _QWORD *v21; // rcx
  __int64 v22; // r11
  __int64 v23; // rdx
  __int64 *result; // rax
  int v25; // edi
  __int64 *v26; // r15
  __int64 *v27; // rcx
  __int64 *v28; // rax
  __int64 *v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 *k; // rdx
  __int64 v35; // rdx
  __int64 *v36; // r9
  int v37; // r10d
  int v38; // r13d
  _QWORD *v39; // r11
  unsigned int v40; // r8d
  __int64 *v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // rdx
  int v44; // ecx
  _QWORD *v45; // r14
  int v46; // [rsp+Ch] [rbp-134h]
  _BYTE v47[304]; // [rsp+10h] [rbp-130h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x10 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v26 = (__int64 *)(a1 + 16);
    v27 = (__int64 *)(a1 + 272);
  }
  else
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v26 = (__int64 *)(a1 + 16);
      v27 = (__int64 *)(a1 + 272);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 16LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (__int64 *)(v6 + v10);
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 32;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = (__int64 *)v6;
              v12 != j;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *j;
            if ( (unsigned __int64)*j <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            j += 2;
            if ( v12 == j )
              return (__int64 *)sub_C7D6A0(v6, v10, 8);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 15;
          }
          else
          {
            v25 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v25 )
              goto LABEL_75;
            v19 = v25 - 1;
          }
          v20 = v19 & (((0xBF58476D1CE4E5B9LL * v17) >> 31) ^ (484763065 * v17));
          v21 = (_QWORD *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v17 != *v21 )
          {
            v46 = 1;
            v45 = 0;
            while ( v22 != -1 )
            {
              if ( v22 == -2 && !v45 )
                v45 = v21;
              v20 = v19 & (v46 + v20);
              v21 = (_QWORD *)(v18 + 16LL * v20);
              v22 = *v21;
              if ( v17 == *v21 )
                goto LABEL_21;
              ++v46;
            }
            if ( v45 )
              v21 = v45;
          }
LABEL_21:
          *v21 = v17;
          v23 = j[1];
          j += 2;
          v21[1] = v23;
        }
        return (__int64 *)sub_C7D6A0(v6, v10, 8);
      }
      v26 = (__int64 *)(a1 + 16);
      v27 = (__int64 *)(a1 + 272);
      v2 = 64;
    }
  }
  v28 = v26;
  v29 = (__int64 *)v47;
  do
  {
    while ( (unsigned __int64)*v28 > 0xFFFFFFFFFFFFFFFDLL )
    {
      v28 += 2;
      if ( v28 == v27 )
        goto LABEL_33;
    }
    if ( v29 )
      *v29 = *v28;
    v30 = v28[1];
    v28 += 2;
    v29 += 2;
    *(v29 - 1) = v30;
  }
  while ( v28 != v27 );
LABEL_33:
  if ( v2 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v31 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v31;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v32 = *(__int64 **)(a1 + 16);
    v33 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v32 = v26;
    v33 = 32;
  }
  for ( k = &v32[v33]; k != v32; v32 += 2 )
  {
    if ( v32 )
      *v32 = -1;
  }
  result = (__int64 *)v47;
  if ( v29 != (__int64 *)v47 )
  {
    do
    {
      while ( 1 )
      {
        v35 = *result;
        if ( (unsigned __int64)*result <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        result += 2;
        if ( v29 == result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v36 = v26;
        v37 = 15;
      }
      else
      {
        v44 = *(_DWORD *)(a1 + 24);
        v36 = *(__int64 **)(a1 + 16);
        if ( !v44 )
        {
LABEL_75:
          MEMORY[0] = 0;
          BUG();
        }
        v37 = v44 - 1;
      }
      v38 = 1;
      v39 = 0;
      v40 = v37 & (((0xBF58476D1CE4E5B9LL * v35) >> 31) ^ (484763065 * v35));
      v41 = &v36[2 * v40];
      v42 = *v41;
      if ( v35 != *v41 )
      {
        while ( v42 != -1 )
        {
          if ( v42 == -2 && !v39 )
            v39 = v41;
          v40 = v37 & (v38 + v40);
          v41 = &v36[2 * v40];
          v42 = *v41;
          if ( v35 == *v41 )
            goto LABEL_48;
          ++v38;
        }
        if ( v39 )
          v41 = v39;
      }
LABEL_48:
      *v41 = v35;
      v43 = result[1];
      result += 2;
      v41[1] = v43;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v29 != result );
  }
  return result;
}
