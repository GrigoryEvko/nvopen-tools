// Function: sub_37C5F80
// Address: 0x37c5f80
//
_BYTE *__fastcall sub_37C5F80(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // r12
  unsigned int v5; // eax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  __int64 v12; // rdi
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  __int64 j; // rax
  int v17; // edx
  __int64 v18; // r10
  int v19; // esi
  int v20; // r15d
  int *v21; // r14
  unsigned int v22; // r8d
  int *v23; // rcx
  int v24; // r11d
  __int64 v25; // rdx
  _BYTE *result; // rax
  int v27; // esi
  _DWORD *v28; // r15
  _DWORD *v29; // rcx
  _DWORD *v30; // rax
  _DWORD *v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rax
  _DWORD *v34; // rax
  __int64 v35; // rdx
  _DWORD *k; // rdx
  int v37; // edx
  _DWORD *v38; // r8
  int v39; // edi
  int v40; // r12d
  int *v41; // r10
  unsigned int v42; // esi
  int *v43; // rcx
  int v44; // r9d
  __int64 v45; // rdx
  int v46; // ecx
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
    v28 = (_DWORD *)(a1 + 16);
    v29 = (_DWORD *)(a1 + 144);
  }
  else
  {
    v5 = sub_AF1560(a2 - 1);
    v2 = v5;
    if ( v5 > 0x40 )
    {
      v28 = (_DWORD *)(a1 + 16);
      v29 = (_DWORD *)(a1 + 144);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 1024;
        v2 = 64;
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
          v13 = *(_DWORD **)(a1 + 16);
          v14 = 4LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 32;
        }
        for ( i = &v13[v14]; i != v13; v13 += 4 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_DWORD *)j;
            if ( *(_DWORD *)j <= 0xFFFFFFFD )
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
            v27 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v27 )
              goto LABEL_74;
            v19 = v27 - 1;
          }
          v20 = 1;
          v21 = 0;
          v22 = v19 & (37 * v17);
          v23 = (int *)(v18 + 16LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (int *)(v18 + 16LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_21;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_21:
          *v23 = v17;
          v25 = *(_QWORD *)(j + 8);
          j += 16;
          *((_QWORD *)v23 + 1) = v25;
        }
        return (_BYTE *)sub_C7D6A0(v6, v10, 8);
      }
      v28 = (_DWORD *)(a1 + 16);
      v29 = (_DWORD *)(a1 + 144);
      v2 = 64;
    }
  }
  v30 = v28;
  v31 = v47;
  do
  {
    while ( *v30 > 0xFFFFFFFD )
    {
      v30 += 4;
      if ( v30 == v29 )
        goto LABEL_33;
    }
    if ( v31 )
      *v31 = *v30;
    v32 = *((_QWORD *)v30 + 1);
    v30 += 4;
    v31 += 4;
    *((_QWORD *)v31 - 1) = v32;
  }
  while ( v30 != v29 );
LABEL_33:
  if ( v2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(_DWORD **)(a1 + 16);
    v35 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v28;
    v35 = 32;
  }
  for ( k = &v34[v35]; k != v34; v34 += 4 )
  {
    if ( v34 )
      *v34 = -1;
  }
  for ( result = v47;
        v31 != (_DWORD *)result;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v37 = *(_DWORD *)result;
      if ( *(_DWORD *)result <= 0xFFFFFFFD )
        break;
      result += 16;
      if ( v31 == (_DWORD *)result )
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
      v38 = *(_DWORD **)(a1 + 16);
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
    v43 = &v38[4 * v42];
    v44 = *v43;
    if ( v37 != *v43 )
    {
      while ( v44 != -1 )
      {
        if ( v44 == -2 && !v41 )
          v41 = v43;
        v42 = v39 & (v40 + v42);
        v43 = &v38[4 * v42];
        v44 = *v43;
        if ( v37 == *v43 )
          goto LABEL_48;
        ++v40;
      }
      if ( v41 )
        v43 = v41;
    }
LABEL_48:
    *v43 = v37;
    v45 = *((_QWORD *)result + 1);
    result += 16;
    *((_QWORD *)v43 + 1) = v45;
  }
  return result;
}
