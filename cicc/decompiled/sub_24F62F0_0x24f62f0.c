// Function: sub_24F62F0
// Address: 0x24f62f0
//
_BYTE *__fastcall sub_24F62F0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
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
  __int64 v17; // r10
  int v18; // edi
  int v19; // r15d
  __int64 *v20; // r14
  unsigned int v21; // esi
  __int64 *v22; // rcx
  __int64 v23; // r11
  __int64 v24; // rdx
  int v25; // edi
  _BYTE *result; // rax
  _QWORD *v27; // r13
  _QWORD *v28; // rcx
  _QWORD *v29; // rax
  _QWORD *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *k; // rdx
  _QWORD *v36; // r8
  int v37; // edi
  int v38; // r14d
  __int64 *v39; // r10
  unsigned int v40; // esi
  __int64 *v41; // rcx
  __int64 v42; // r9
  int v43; // edi
  _BYTE v44[176]; // [rsp+10h] [rbp-B0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v27 = (_QWORD *)(a1 + 16);
    v28 = (_QWORD *)(a1 + 144);
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v27 = (_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(a1 + 144);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 16LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v4 + v10;
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
            *v13 = -4096;
        }
        for ( j = v4; v12 != j; j += 16 )
        {
          v24 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v24 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = a1 + 16;
              v18 = 7;
            }
            else
            {
              v25 = *(_DWORD *)(a1 + 24);
              v17 = *(_QWORD *)(a1 + 16);
              if ( !v25 )
                goto LABEL_77;
              v18 = v25 - 1;
            }
            v19 = 1;
            v20 = 0;
            v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v22 = (__int64 *)(v17 + 16LL * v21);
            v23 = *v22;
            if ( *v22 != v24 )
            {
              while ( v23 != -4096 )
              {
                if ( v23 == -8192 && !v20 )
                  v20 = v22;
                v21 = v18 & (v19 + v21);
                v22 = (__int64 *)(v17 + 16LL * v21);
                v23 = *v22;
                if ( v24 == *v22 )
                  goto LABEL_16;
                ++v19;
              }
              if ( v20 )
                v22 = v20;
            }
LABEL_16:
            *v22 = v24;
            *((_DWORD *)v22 + 2) = *(_DWORD *)(j + 8);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return (_BYTE *)sub_C7D6A0(v4, v10, 8);
      }
      v27 = (_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(a1 + 144);
      v2 = 64;
    }
  }
  v29 = v27;
  v30 = v44;
  do
  {
    v31 = *v29;
    if ( *v29 != -4096 && v31 != -8192 )
    {
      if ( v30 )
        *v30 = v31;
      v30 += 2;
      *((_DWORD *)v30 - 2) = *((_DWORD *)v29 + 2);
    }
    v29 += 2;
  }
  while ( v29 != v28 );
  if ( v2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v27;
    v34 = 16;
  }
  for ( k = &v33[v34]; k != v33; v33 += 2 )
  {
    if ( v33 )
      *v33 = -4096;
  }
  for ( result = v44; v30 != (_QWORD *)result; result += 16 )
  {
    v24 = *(_QWORD *)result;
    if ( *(_QWORD *)result != -4096 && v24 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v36 = v27;
        v37 = 7;
      }
      else
      {
        v43 = *(_DWORD *)(a1 + 24);
        v36 = *(_QWORD **)(a1 + 16);
        if ( !v43 )
        {
LABEL_77:
          MEMORY[0] = v24;
          BUG();
        }
        v37 = v43 - 1;
      }
      v38 = 1;
      v39 = 0;
      v40 = v37 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v41 = &v36[2 * v40];
      v42 = *v41;
      if ( v24 != *v41 )
      {
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v39 )
            v39 = v41;
          v40 = v37 & (v38 + v40);
          v41 = &v36[2 * v40];
          v42 = *v41;
          if ( v24 == *v41 )
            goto LABEL_47;
          ++v38;
        }
        if ( v39 )
          v41 = v39;
      }
LABEL_47:
      *v41 = v24;
      *((_DWORD *)v41 + 2) = *((_DWORD *)result + 2);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
