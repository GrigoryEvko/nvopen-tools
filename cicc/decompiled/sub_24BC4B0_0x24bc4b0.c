// Function: sub_24BC4B0
// Address: 0x24bc4b0
//
__int64 *__fastcall sub_24BC4B0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r8
  bool v11; // zf
  __int64 *v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v17; // rdx
  __int64 v18; // r10
  int v19; // esi
  unsigned int v20; // ecx
  _QWORD *v21; // r9
  __int64 v22; // r11
  __int64 *result; // rax
  int v24; // esi
  __int64 *v25; // r15
  __int64 *v26; // rcx
  __int64 *v27; // rax
  __int64 *v28; // r13
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 *k; // rdx
  __int64 v33; // rdx
  __int64 *v34; // r9
  int v35; // r8d
  int v36; // r12d
  _QWORD *v37; // r11
  int v38; // ecx
  __int64 *v39; // rsi
  __int64 v40; // r10
  int v41; // ecx
  _QWORD *v42; // r14
  int v43; // [rsp+Ch] [rbp-B4h]
  _BYTE v44[176]; // [rsp+10h] [rbp-B0h] BYREF

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
    v25 = (__int64 *)(a1 + 16);
    v26 = (__int64 *)(a1 + 144);
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
      v25 = (__int64 *)(a1 + 16);
      v26 = (__int64 *)(a1 + 144);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 8LL * (unsigned int)v5;
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
        v8 = 512;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 8LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (__int64 *)(v6 + v10);
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 16;
        }
        for ( i = &v13[v14]; i != v13; ++v13 )
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
            if ( v12 == ++j )
              return (__int64 *)sub_C7D6A0(v6, v10, 8);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 15;
          }
          else
          {
            v24 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v24 )
              goto LABEL_75;
            v19 = v24 - 1;
          }
          v20 = v19 & (((0xBF58476D1CE4E5B9LL * v17) >> 31) ^ (484763065 * v17));
          v21 = (_QWORD *)(v18 + 8LL * v20);
          v22 = *v21;
          if ( v17 != *v21 )
          {
            v43 = 1;
            v42 = 0;
            while ( v22 != -1 )
            {
              if ( v22 == -2 && !v42 )
                v42 = v21;
              v20 = v19 & (v43 + v20);
              v21 = (_QWORD *)(v18 + 8LL * v20);
              v22 = *v21;
              if ( v17 == *v21 )
                goto LABEL_21;
              ++v43;
            }
            if ( v42 )
              v21 = v42;
          }
LABEL_21:
          *v21 = v17;
          ++j;
        }
        return (__int64 *)sub_C7D6A0(v6, v10, 8);
      }
      v25 = (__int64 *)(a1 + 16);
      v26 = (__int64 *)(a1 + 144);
      v2 = 64;
    }
  }
  v27 = v25;
  v28 = (__int64 *)v44;
  do
  {
    while ( (unsigned __int64)*v27 > 0xFFFFFFFFFFFFFFFDLL )
    {
      if ( ++v27 == v26 )
        goto LABEL_33;
    }
    if ( v28 )
      *v28 = *v27;
    ++v27;
    ++v28;
  }
  while ( v27 != v26 );
LABEL_33:
  if ( v2 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v29 = sub_C7D670(8LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v29;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v30 = *(__int64 **)(a1 + 16);
    v31 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v30 = v25;
    v31 = 16;
  }
  for ( k = &v30[v31]; k != v30; ++v30 )
  {
    if ( v30 )
      *v30 = -1;
  }
  result = (__int64 *)v44;
  if ( v28 != (__int64 *)v44 )
  {
    do
    {
      while ( 1 )
      {
        v33 = *result;
        if ( (unsigned __int64)*result <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        if ( v28 == ++result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v34 = v25;
        v35 = 15;
      }
      else
      {
        v41 = *(_DWORD *)(a1 + 24);
        v34 = *(__int64 **)(a1 + 16);
        if ( !v41 )
        {
LABEL_75:
          MEMORY[0] = 0;
          BUG();
        }
        v35 = v41 - 1;
      }
      v36 = 1;
      v37 = 0;
      v38 = v35 & (((0xBF58476D1CE4E5B9LL * v33) >> 31) ^ (484763065 * v33));
      v39 = &v34[v38];
      v40 = *v39;
      if ( v33 != *v39 )
      {
        while ( v40 != -1 )
        {
          if ( v40 == -2 && !v37 )
            v37 = v39;
          v38 = v35 & (v36 + v38);
          v39 = &v34[v38];
          v40 = *v39;
          if ( v33 == *v39 )
            goto LABEL_48;
          ++v36;
        }
        if ( v37 )
          v39 = v37;
      }
LABEL_48:
      *v39 = v33;
      ++result;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v28 != result );
  }
  return result;
}
