// Function: sub_36F0660
// Address: 0x36f0660
//
unsigned __int8 *__fastcall sub_36F0660(__int64 a1, unsigned int a2)
{
  char v4; // si
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  __int64 v12; // rsi
  _BYTE *v13; // rax
  __int64 v14; // rdx
  _BYTE *i; // rdx
  __int64 j; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // r10
  int v19; // ecx
  int v20; // r15d
  char *v21; // r14
  unsigned int v22; // r8d
  unsigned __int8 *v23; // rdi
  unsigned __int8 v24; // r11
  int v25; // edx
  unsigned __int8 *result; // rax
  int v27; // ecx
  unsigned __int8 *v28; // r15
  unsigned __int8 *v29; // rcx
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // r13
  int v32; // edx
  __int64 v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  unsigned __int8 *k; // rdx
  unsigned __int8 v37; // dl
  unsigned __int8 *v38; // r9
  int v39; // edi
  int v40; // r11d
  char *v41; // r10
  unsigned int v42; // esi
  unsigned __int8 *v43; // rcx
  unsigned __int8 v44; // r8
  int v45; // edx
  int v46; // ecx
  _BYTE v47[112]; // [rsp+10h] [rbp-70h] BYREF

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
    v28 = (unsigned __int8 *)(a1 + 16);
    v29 = (unsigned __int8 *)(a1 + 80);
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
    a2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v28 = (unsigned __int8 *)(a1 + 16);
      v29 = (unsigned __int8 *)(a1 + 80);
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
        a2 = 64;
        v8 = 512;
LABEL_5:
        v9 = sub_C7D670(v8, 4);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 8LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v6 + v10;
        if ( v11 )
        {
          v13 = *(_BYTE **)(a1 + 16);
          v14 = 8LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_BYTE *)(a1 + 16);
          v14 = 64;
        }
        for ( i = &v13[v14]; i != v13; v13 += 8 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_BYTE *)j;
            if ( *(_BYTE *)j <= 0xFDu )
              break;
            j += 8;
            if ( v12 == j )
              return (unsigned __int8 *)sub_C7D6A0(v6, v10, 4);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 7;
          }
          else
          {
            v27 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v27 )
              goto LABEL_74;
            v19 = v27 - 1;
          }
          v20 = 1;
          v21 = 0;
          v22 = v19 & (37 * v17);
          v23 = (unsigned __int8 *)(v18 + 8LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != 0xFF )
            {
              if ( v24 == 0xFE && !v21 )
                v21 = (char *)v23;
              v22 = v19 & (v20 + v22);
              v23 = (unsigned __int8 *)(v18 + 8LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_21;
              ++v20;
            }
            if ( v21 )
              v23 = (unsigned __int8 *)v21;
          }
LABEL_21:
          *v23 = v17;
          v25 = *(_DWORD *)(j + 4);
          j += 8;
          *((_DWORD *)v23 + 1) = v25;
        }
        return (unsigned __int8 *)sub_C7D6A0(v6, v10, 4);
      }
      v28 = (unsigned __int8 *)(a1 + 16);
      v29 = (unsigned __int8 *)(a1 + 80);
      a2 = 64;
    }
  }
  v30 = v28;
  v31 = v47;
  do
  {
    while ( *v30 > 0xFDu )
    {
      v30 += 8;
      if ( v30 == v29 )
        goto LABEL_33;
    }
    if ( v31 )
      *v31 = *v30;
    v32 = *((_DWORD *)v30 + 1);
    v30 += 8;
    v31 += 8;
    *((_DWORD *)v31 - 1) = v32;
  }
  while ( v30 != v29 );
LABEL_33:
  if ( a2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(8LL * a2, 4);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(unsigned __int8 **)(a1 + 16);
    v35 = 8LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v28;
    v35 = 64;
  }
  for ( k = &v34[v35]; k != v34; v34 += 8 )
  {
    if ( v34 )
      *v34 = -1;
  }
  for ( result = v47; v31 != result; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2)
                                                         | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v37 = *result;
      if ( *result <= 0xFDu )
        break;
      result += 8;
      if ( v31 == result )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v38 = v28;
      v39 = 7;
    }
    else
    {
      v46 = *(_DWORD *)(a1 + 24);
      v38 = *(unsigned __int8 **)(a1 + 16);
      if ( !v46 )
      {
LABEL_74:
        MEMORY[0] = 0;
        BUG();
      }
      v39 = v46 - 1;
    }
    v40 = 1;
    v41 = 0;
    v42 = v39 & (37 * v37);
    v43 = &v38[8 * v42];
    v44 = *v43;
    if ( v37 != *v43 )
    {
      while ( v44 != 0xFF )
      {
        if ( v44 == 0xFE && !v41 )
          v41 = (char *)v43;
        v42 = v39 & (v40 + v42);
        v43 = &v38[8 * v42];
        v44 = *v43;
        if ( v37 == *v43 )
          goto LABEL_48;
        ++v40;
      }
      if ( v41 )
        v43 = (unsigned __int8 *)v41;
    }
LABEL_48:
    *v43 = v37;
    v45 = *((_DWORD *)result + 1);
    result += 8;
    *((_DWORD *)v43 + 1) = v45;
  }
  return result;
}
