// Function: sub_2BB5410
// Address: 0x2bb5410
//
_BYTE *__fastcall sub_2BB5410(__int64 a1, unsigned int a2)
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
  __int64 v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rax
  __int64 v17; // rdx
  __int64 v18; // r10
  int v19; // edi
  unsigned int v20; // esi
  __int64 *v21; // rcx
  __int64 v22; // r11
  int v23; // edx
  _BYTE *result; // rax
  int v25; // edi
  _QWORD *v26; // r15
  _QWORD *v27; // rcx
  _QWORD *v28; // rax
  _QWORD *v29; // r13
  int v30; // edx
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *k; // rdx
  __int64 v35; // rdx
  _QWORD *v36; // r9
  int v37; // r10d
  int v38; // r12d
  __int64 *v39; // r11
  unsigned int v40; // r8d
  __int64 *v41; // rcx
  __int64 v42; // rsi
  int v43; // edx
  int v44; // ecx
  __int64 *v45; // r14
  int v46; // [rsp+Ch] [rbp-B4h]
  _BYTE v47[176]; // [rsp+10h] [rbp-B0h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v26 = (_QWORD *)(a1 + 16);
    v27 = (_QWORD *)(a1 + 144);
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
      v26 = (_QWORD *)(a1 + 16);
      v27 = (_QWORD *)(a1 + 144);
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
        v12 = v6 + v10;
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 16;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_QWORD *)j;
            if ( *(_QWORD *)j <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            j += 16;
            if ( v12 == j )
              return (_BYTE *)sub_C7D6A0(v6, v10, 8);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 7;
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
          v21 = (__int64 *)(v18 + 16LL * v20);
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
              v21 = (__int64 *)(v18 + 16LL * v20);
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
          v23 = *(_DWORD *)(j + 8);
          j += 16;
          *((_DWORD *)v21 + 2) = v23;
        }
        return (_BYTE *)sub_C7D6A0(v6, v10, 8);
      }
      v26 = (_QWORD *)(a1 + 16);
      v27 = (_QWORD *)(a1 + 144);
      v2 = 64;
    }
  }
  v28 = v26;
  v29 = v47;
  do
  {
    while ( *v28 > 0xFFFFFFFFFFFFFFFDLL )
    {
      v28 += 2;
      if ( v28 == v27 )
        goto LABEL_33;
    }
    if ( v29 )
      *v29 = *v28;
    v30 = *((_DWORD *)v28 + 2);
    v28 += 2;
    v29 += 2;
    *((_DWORD *)v29 - 2) = v30;
  }
  while ( v28 != v27 );
LABEL_33:
  if ( v2 > 8 )
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
    v32 = *(_QWORD **)(a1 + 16);
    v33 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v32 = v26;
    v33 = 16;
  }
  for ( k = &v32[v33]; k != v32; v32 += 2 )
  {
    if ( v32 )
      *v32 = -1;
  }
  for ( result = v47;
        v29 != (_QWORD *)result;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v35 = *(_QWORD *)result;
      if ( *(_QWORD *)result <= 0xFFFFFFFFFFFFFFFDLL )
        break;
      result += 16;
      if ( v29 == (_QWORD *)result )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v36 = v26;
      v37 = 7;
    }
    else
    {
      v44 = *(_DWORD *)(a1 + 24);
      v36 = *(_QWORD **)(a1 + 16);
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
    v43 = *((_DWORD *)result + 2);
    result += 16;
    *((_DWORD *)v41 + 2) = v43;
  }
  return result;
}
