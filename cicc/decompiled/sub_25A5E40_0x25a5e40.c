// Function: sub_25A5E40
// Address: 0x25a5e40
//
_BYTE *__fastcall sub_25A5E40(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r13
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
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r8
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rax
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
  unsigned int v41; // esi
  __int64 *v42; // rcx
  __int64 v43; // r9
  unsigned __int64 v44; // rdx
  int v45; // ecx
  _BYTE v46[688]; // [rsp+0h] [rbp-2B0h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x10 )
  {
    v10 = (_QWORD *)(a1 + 16);
    v11 = (_QWORD *)(a1 + 656);
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
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
      v10 = (_QWORD *)(a1 + 16);
      v11 = (_QWORD *)(a1 + 656);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 40LL * (unsigned int)v5;
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
        v8 = 2560;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = 40LL * v7;
        v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v14 = v6 + v12;
        if ( v13 )
        {
          v15 = *(_QWORD **)(a1 + 16);
          v16 = 5LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v15 = (_QWORD *)(a1 + 16);
          v16 = 80;
        }
        for ( i = &v15[v16]; i != v15; v15 += 5 )
        {
          if ( v15 )
            *v15 = -2;
        }
        for ( j = v6; v14 != j; j += 40 )
        {
          v27 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -2 && v27 != -16 )
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
            v23 = v20 & (v27 ^ (v27 >> 9));
            v24 = (__int64 *)(v19 + 40LL * v23);
            v25 = *v24;
            if ( *v24 != v27 )
            {
              while ( v25 != -2 )
              {
                if ( !v22 && v25 == -16 )
                  v22 = v24;
                v23 = v20 & (v21 + v23);
                v24 = (__int64 *)(v19 + 40LL * v23);
                v25 = *v24;
                if ( v27 == *v24 )
                  goto LABEL_18;
                ++v21;
              }
              if ( v22 )
                v24 = v22;
            }
LABEL_18:
            *v24 = *(_QWORD *)j;
            *((_DWORD *)v24 + 2) = *(_DWORD *)(j + 8);
            v24[2] = *(_QWORD *)(j + 16);
            v24[3] = *(_QWORD *)(j + 24);
            v24[4] = *(_QWORD *)(j + 32);
            *(_QWORD *)(j + 16) = 0;
            *(_QWORD *)(j + 32) = 0;
            *(_QWORD *)(j + 24) = 0;
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v26 = *(_QWORD *)(j + 16);
            if ( v26 )
              j_j___libc_free_0(v26);
          }
        }
        return (_BYTE *)sub_C7D6A0(v6, v12, 8);
      }
      v10 = (_QWORD *)(a1 + 16);
      v11 = (_QWORD *)(a1 + 656);
      v2 = 64;
    }
  }
  v29 = v10;
  v30 = v46;
  do
  {
    v31 = *v29;
    if ( *v29 != -2 && v31 != -16 )
    {
      if ( v30 )
        *v30 = v31;
      v30 += 5;
      *((_DWORD *)v30 - 8) = *((_DWORD *)v29 + 2);
      *(v30 - 3) = v29[2];
      *(v30 - 2) = v29[3];
      *(v30 - 1) = v29[4];
    }
    v29 += 5;
  }
  while ( v29 != v11 );
  if ( v2 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v32 = sub_C7D670(40LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v32;
  }
  v13 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v13 )
  {
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = v10;
    v34 = 80;
  }
  for ( k = &v33[v34]; k != v33; v33 += 5 )
  {
    if ( v33 )
      *v33 = -2;
  }
  for ( result = v46; v30 != (_QWORD *)result; result += 40 )
  {
    v44 = *(_QWORD *)result;
    if ( *(_QWORD *)result != -16 && v44 != -2 )
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
      v41 = v38 & (v44 ^ (v44 >> 9));
      v42 = &v37[5 * v41];
      v43 = *v42;
      if ( v44 != *v42 )
      {
        while ( v43 != -2 )
        {
          if ( v43 == -16 && !v40 )
            v40 = v42;
          v41 = v38 & (v39 + v41);
          v42 = &v37[5 * v41];
          v43 = *v42;
          if ( v44 == *v42 )
            goto LABEL_46;
          ++v39;
        }
        if ( v40 )
          v42 = v40;
      }
LABEL_46:
      *v42 = *(_QWORD *)result;
      *((_DWORD *)v42 + 2) = *((_DWORD *)result + 2);
      v42[2] = *((_QWORD *)result + 2);
      v42[3] = *((_QWORD *)result + 3);
      v42[4] = *((_QWORD *)result + 4);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
