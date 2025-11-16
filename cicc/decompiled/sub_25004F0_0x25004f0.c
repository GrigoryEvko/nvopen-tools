// Function: sub_25004F0
// Address: 0x25004f0
//
__int64 *__fastcall sub_25004F0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r10
  bool v11; // zf
  __int64 *v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v17; // rdx
  __int64 v18; // r11
  int v19; // esi
  unsigned int v20; // r9d
  __int64 *v21; // rcx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 *result; // rax
  int v25; // esi
  __int64 *v26; // r13
  __int64 *v27; // rcx
  __int64 *v28; // rax
  __int64 *v29; // r12
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *k; // rdx
  __int64 v34; // rdx
  __int64 *v35; // r9
  int v36; // esi
  unsigned int v37; // r8d
  __int64 *v38; // rcx
  __int64 v39; // r10
  __int64 v40; // rdx
  int v41; // esi
  int v42; // r15d
  __int64 *v43; // r14
  int v44; // ecx
  int v45; // [rsp+4h] [rbp-7Ch]
  __int64 *v46; // [rsp+8h] [rbp-78h]
  _BYTE v47[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v26 = (__int64 *)(a1 + 16);
    v27 = (__int64 *)(a1 + 80);
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
      v27 = (__int64 *)(a1 + 80);
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
          v14 = 8;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = 0x7FFFFFFFFFFFFFFFLL;
        }
        for ( j = (__int64 *)v6;
              v12 != j;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *j;
            if ( *j <= 0x7FFFFFFFFFFFFFFDLL )
              break;
            j += 2;
            if ( v12 == j )
              return (__int64 *)sub_C7D6A0(v6, v10, 8);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 3;
          }
          else
          {
            v25 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v25 )
              goto LABEL_77;
            v19 = v25 - 1;
          }
          v20 = v19 & (37 * v17);
          v21 = (__int64 *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v17 != *v21 )
          {
            v45 = 1;
            v46 = 0;
            while ( v22 != 0x7FFFFFFFFFFFFFFFLL )
            {
              if ( v22 == 0x7FFFFFFFFFFFFFFELL )
              {
                if ( v46 )
                  v21 = v46;
                v46 = v21;
              }
              v20 = v19 & (v45 + v20);
              v21 = (__int64 *)(v18 + 16LL * v20);
              v22 = *v21;
              if ( v17 == *v21 )
                goto LABEL_21;
              ++v45;
            }
            if ( v46 )
              v21 = v46;
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
      v27 = (__int64 *)(a1 + 80);
      v2 = 64;
    }
  }
  v28 = v26;
  v29 = (__int64 *)v47;
  do
  {
    if ( *v28 <= 0x7FFFFFFFFFFFFFFDLL )
    {
      if ( v29 )
        *v29 = *v28;
      v29 += 2;
      *(v29 - 1) = v28[1];
    }
    v28 += 2;
  }
  while ( v28 != v27 );
  if ( v2 > 4 )
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
    v31 = *(__int64 **)(a1 + 16);
    v32 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v31 = v26;
    v32 = 8;
  }
  for ( k = &v31[v32]; k != v31; v31 += 2 )
  {
    if ( v31 )
      *v31 = 0x7FFFFFFFFFFFFFFFLL;
  }
  result = (__int64 *)v47;
  if ( v29 != (__int64 *)v47 )
  {
    do
    {
      while ( 1 )
      {
        v34 = *result;
        if ( *result <= 0x7FFFFFFFFFFFFFFDLL )
          break;
        result += 2;
        if ( v29 == result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v35 = v26;
        v36 = 3;
      }
      else
      {
        v41 = *(_DWORD *)(a1 + 24);
        v35 = *(__int64 **)(a1 + 16);
        if ( !v41 )
        {
LABEL_77:
          MEMORY[0] = 0;
          BUG();
        }
        v36 = v41 - 1;
      }
      v37 = v36 & (37 * v34);
      v38 = &v35[2 * v37];
      v39 = *v38;
      if ( v34 != *v38 )
      {
        v42 = 1;
        v43 = 0;
        while ( v39 != 0x7FFFFFFFFFFFFFFFLL )
        {
          if ( v39 == 0x7FFFFFFFFFFFFFFELL && !v43 )
            v43 = v38;
          v44 = v42++;
          v37 = v36 & (v44 + v37);
          v38 = &v35[2 * v37];
          v39 = *v38;
          if ( v34 == *v38 )
            goto LABEL_45;
        }
        if ( v43 )
          v38 = v43;
      }
LABEL_45:
      *v38 = v34;
      v40 = result[1];
      result += 2;
      v38[1] = v40;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v29 != result );
  }
  return result;
}
