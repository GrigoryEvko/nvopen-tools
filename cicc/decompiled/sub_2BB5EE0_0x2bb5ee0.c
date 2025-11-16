// Function: sub_2BB5EE0
// Address: 0x2bb5ee0
//
_BYTE *__fastcall sub_2BB5EE0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r14d
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
  _QWORD *v26; // r12
  _QWORD *v27; // rcx
  _QWORD *v28; // rax
  _QWORD *v29; // r14
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *k; // rdx
  __int64 v34; // rdx
  _QWORD *v35; // r9
  int v36; // r10d
  int v37; // r13d
  __int64 *v38; // r11
  unsigned int v39; // r8d
  __int64 *v40; // rcx
  __int64 v41; // rsi
  int v42; // edx
  int v43; // ecx
  __int64 *v44; // r14
  int v45; // [rsp+Ch] [rbp-54h]
  _BYTE v46[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 2 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v26 = (_QWORD *)(a1 + 16);
    v27 = (_QWORD *)(a1 + 48);
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
      v27 = (_QWORD *)(a1 + 48);
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
          v14 = 4;
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
            v19 = 1;
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
            v45 = 1;
            v44 = 0;
            while ( v22 != -1 )
            {
              if ( v22 == -2 && !v44 )
                v44 = v21;
              v20 = v19 & (v45 + v20);
              v21 = (__int64 *)(v18 + 16LL * v20);
              v22 = *v21;
              if ( v17 == *v21 )
                goto LABEL_21;
              ++v45;
            }
            if ( v44 )
              v21 = v44;
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
      v27 = (_QWORD *)(a1 + 48);
      v2 = 64;
    }
  }
  v28 = v26;
  v29 = v46;
  do
  {
    if ( *v28 <= 0xFFFFFFFFFFFFFFFDLL )
    {
      if ( v29 )
        *v29 = *v28;
      v29 += 2;
      *((_DWORD *)v29 - 2) = *((_DWORD *)v28 + 2);
    }
    v28 += 2;
  }
  while ( v28 != v27 );
  if ( v2 > 2 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v30 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v30;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v31 = *(_QWORD **)(a1 + 16);
    v32 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v31 = v26;
    v32 = 4;
  }
  for ( k = &v31[v32]; k != v31; v31 += 2 )
  {
    if ( v31 )
      *v31 = -1;
  }
  for ( result = v46;
        v29 != (_QWORD *)result;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v34 = *(_QWORD *)result;
      if ( *(_QWORD *)result <= 0xFFFFFFFFFFFFFFFDLL )
        break;
      result += 16;
      if ( v29 == (_QWORD *)result )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = v26;
      v36 = 1;
    }
    else
    {
      v43 = *(_DWORD *)(a1 + 24);
      v35 = *(_QWORD **)(a1 + 16);
      if ( !v43 )
      {
LABEL_75:
        MEMORY[0] = 0;
        BUG();
      }
      v36 = v43 - 1;
    }
    v37 = 1;
    v38 = 0;
    v39 = v36 & (((0xBF58476D1CE4E5B9LL * v34) >> 31) ^ (484763065 * v34));
    v40 = &v35[2 * v39];
    v41 = *v40;
    if ( v34 != *v40 )
    {
      while ( v41 != -1 )
      {
        if ( v41 == -2 && !v38 )
          v38 = v40;
        v39 = v36 & (v37 + v39);
        v40 = &v35[2 * v39];
        v41 = *v40;
        if ( v34 == *v40 )
          goto LABEL_45;
        ++v37;
      }
      if ( v38 )
        v40 = v38;
    }
LABEL_45:
    *v40 = v34;
    v42 = *((_DWORD *)result + 2);
    result += 16;
    *((_DWORD *)v40 + 2) = v42;
  }
  return result;
}
