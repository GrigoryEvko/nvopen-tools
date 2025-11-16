// Function: sub_DB20A0
// Address: 0xdb20a0
//
_BYTE *__fastcall sub_DB20A0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r13
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD *v10; // r14
  _QWORD *v11; // rcx
  __int64 v12; // r14
  bool v13; // zf
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v19; // rdi
  int v20; // esi
  int v21; // r10d
  __int64 *v22; // r9
  unsigned int v23; // edx
  __int64 *v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdi
  __int64 v27; // rax
  int v28; // edx
  _QWORD *v29; // rax
  _QWORD *v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *k; // rdx
  _BYTE *result; // rax
  _QWORD *v37; // r8
  int v38; // edi
  int v39; // r11d
  __int64 *v40; // r10
  unsigned int v41; // ecx
  __int64 *v42; // rsi
  __int64 v43; // r9
  __int64 v44; // rdx
  int v45; // ecx
  _BYTE v46[432]; // [rsp+0h] [rbp-1B0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x10 )
  {
    v10 = (_QWORD *)(a1 + 16);
    v11 = (_QWORD *)(a1 + 400);
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
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
      v10 = (_QWORD *)(a1 + 16);
      v11 = (_QWORD *)(a1 + 400);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 24LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 1536;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = 24LL * v7;
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v14 = v4 + v12;
        if ( v13 )
        {
          v15 = *(_QWORD **)(a1 + 16);
          v16 = 3LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v15 = (_QWORD *)(a1 + 16);
          v16 = 48;
        }
        for ( i = &v15[v16]; i != v15; v15 += 3 )
        {
          if ( v15 )
            *v15 = -4096;
        }
        for ( j = v4; v14 != j; j += 24 )
        {
          v27 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v27 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v19 = a1 + 16;
              v20 = 15;
            }
            else
            {
              v28 = *(_DWORD *)(a1 + 24);
              v19 = *(_QWORD *)(a1 + 16);
              if ( !v28 )
              {
                MEMORY[0] = *(_QWORD *)j;
                BUG();
              }
              v20 = v28 - 1;
            }
            v21 = 1;
            v22 = 0;
            v23 = v20 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v24 = (__int64 *)(v19 + 24LL * v23);
            v25 = *v24;
            if ( *v24 != v27 )
            {
              while ( v25 != -4096 )
              {
                if ( !v22 && v25 == -8192 )
                  v22 = v24;
                v23 = v20 & (v21 + v23);
                v24 = (__int64 *)(v19 + 24LL * v23);
                v25 = *v24;
                if ( v27 == *v24 )
                  goto LABEL_18;
                ++v21;
              }
              if ( v22 )
                v24 = v22;
            }
LABEL_18:
            *v24 = v27;
            *((_DWORD *)v24 + 4) = *(_DWORD *)(j + 16);
            v24[1] = *(_QWORD *)(j + 8);
            *(_DWORD *)(j + 16) = 0;
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            if ( *(_DWORD *)(j + 16) > 0x40u )
            {
              v26 = *(_QWORD *)(j + 8);
              if ( v26 )
                j_j___libc_free_0_0(v26);
            }
          }
        }
        return (_BYTE *)sub_C7D6A0(v4, v12, 8);
      }
      v10 = (_QWORD *)(a1 + 16);
      v11 = (_QWORD *)(a1 + 400);
      v2 = 64;
    }
  }
  v29 = v10;
  v30 = v46;
  do
  {
    v31 = *v29;
    if ( *v29 != -4096 && v31 != -8192 )
    {
      if ( v30 )
        *v30 = v31;
      v30 += 3;
      *((_DWORD *)v30 - 2) = *((_DWORD *)v29 + 4);
      *(v30 - 2) = v29[1];
    }
    v29 += 3;
  }
  while ( v29 != v11 );
  if ( v2 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(24LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v13 )
  {
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v10;
    v34 = 48;
  }
  for ( k = &v33[v34]; k != v33; v33 += 3 )
  {
    if ( v33 )
      *v33 = -4096;
  }
  for ( result = v46; v30 != (_QWORD *)result; result += 24 )
  {
    v44 = *(_QWORD *)result;
    if ( *(_QWORD *)result != -8192 && v44 != -4096 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v37 = v10;
        v38 = 15;
      }
      else
      {
        v45 = *(_DWORD *)(a1 + 24);
        v37 = *(_QWORD **)(a1 + 16);
        if ( !v45 )
        {
          MEMORY[0] = *(_QWORD *)result;
          BUG();
        }
        v38 = v45 - 1;
      }
      v39 = 1;
      v40 = 0;
      v41 = v38 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v42 = &v37[3 * v41];
      v43 = *v42;
      if ( v44 != *v42 )
      {
        while ( v43 != -4096 )
        {
          if ( v43 == -8192 && !v40 )
            v40 = v42;
          v41 = v38 & (v39 + v41);
          v42 = &v37[3 * v41];
          v43 = *v42;
          if ( v44 == *v42 )
            goto LABEL_47;
          ++v39;
        }
        if ( v40 )
          v42 = v40;
      }
LABEL_47:
      *v42 = v44;
      *((_DWORD *)v42 + 4) = *((_DWORD *)result + 4);
      v42[1] = *((_QWORD *)result + 1);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
