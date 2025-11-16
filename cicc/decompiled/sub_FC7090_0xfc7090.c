// Function: sub_FC7090
// Address: 0xfc7090
//
_BYTE *__fastcall sub_FC7090(__int64 a1, unsigned int a2)
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
  __int64 v20; // rsi
  int v21; // r10d
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rax
  int v29; // edx
  _QWORD *v30; // rax
  _QWORD *v31; // rbx
  __int64 v32; // rdx
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
  __int64 v44; // rdx
  int v45; // ecx
  __int64 v46; // rax
  _BYTE v47[816]; // [rsp+0h] [rbp-330h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x20 )
  {
    v10 = (_QWORD *)(a1 + 16);
    v11 = (_QWORD *)(a1 + 784);
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
      v11 = (_QWORD *)(a1 + 784);
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
          v16 = 96;
        }
        for ( i = &v15[v16]; i != v15; v15 += 3 )
        {
          if ( v15 )
            *v15 = -4096;
        }
        for ( j = v4; v14 != j; j += 24 )
        {
          v28 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v28 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v19 = a1 + 16;
              v20 = 31;
            }
            else
            {
              v29 = *(_DWORD *)(a1 + 24);
              v19 = *(_QWORD *)(a1 + 16);
              if ( !v29 )
              {
                MEMORY[0] = *(_QWORD *)j;
                BUG();
              }
              v20 = (unsigned int)(v29 - 1);
            }
            v21 = 1;
            v22 = 0;
            v23 = (unsigned int)v20 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v24 = v19 + 24 * v23;
            v25 = *(_QWORD *)v24;
            if ( *(_QWORD *)v24 != v28 )
            {
              while ( v25 != -4096 )
              {
                if ( v25 == -8192 && !v22 )
                  v22 = v24;
                v23 = (unsigned int)v20 & (v21 + (_DWORD)v23);
                v24 = v19 + 24LL * (unsigned int)v23;
                v25 = *(_QWORD *)v24;
                if ( v28 == *(_QWORD *)v24 )
                  goto LABEL_18;
                ++v21;
              }
              if ( v22 )
                v24 = v22;
            }
LABEL_18:
            *(_QWORD *)v24 = v28;
            *(_BYTE *)(v24 + 8) = *(_BYTE *)(j + 8);
            *(_DWORD *)(v24 + 12) = *(_DWORD *)(j + 12);
            *(_QWORD *)(v24 + 16) = *(_QWORD *)(j + 16);
            *(_QWORD *)(j + 16) = 0;
            v26 = (unsigned int)(2 * (*(_DWORD *)(a1 + 8) >> 1) + 2);
            *(_DWORD *)(a1 + 8) = v26 | *(_DWORD *)(a1 + 8) & 1;
            v27 = *(_QWORD *)(j + 16);
            if ( v27 )
              sub_BA65D0(v27, v20, v26, v23, v25);
          }
        }
        return (_BYTE *)sub_C7D6A0(v4, v12, 8);
      }
      v10 = (_QWORD *)(a1 + 16);
      v11 = (_QWORD *)(a1 + 784);
      v2 = 64;
    }
  }
  v30 = v10;
  v31 = v47;
  do
  {
    v32 = *v30;
    if ( *v30 != -4096 && v32 != -8192 )
    {
      if ( v31 )
        *v31 = v32;
      v31 += 3;
      *((_BYTE *)v31 - 16) = *((_BYTE *)v30 + 8);
      *((_DWORD *)v31 - 3) = *((_DWORD *)v30 + 3);
      *(v31 - 1) = v30[2];
    }
    v30 += 3;
  }
  while ( v30 != v11 );
  if ( v2 > 0x20 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v46 = sub_C7D670(24LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v46;
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
    v34 = 96;
  }
  for ( k = &v33[v34]; k != v33; v33 += 3 )
  {
    if ( v33 )
      *v33 = -4096;
  }
  for ( result = v47; v31 != (_QWORD *)result; result += 24 )
  {
    v44 = *(_QWORD *)result;
    if ( *(_QWORD *)result != -4096 && v44 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v37 = v10;
        v38 = 31;
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
            goto LABEL_46;
          ++v39;
        }
        if ( v40 )
          v42 = v40;
      }
LABEL_46:
      *v42 = v44;
      *((_BYTE *)v42 + 8) = result[8];
      *((_DWORD *)v42 + 3) = *((_DWORD *)result + 3);
      v42[2] = *((_QWORD *)result + 2);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
