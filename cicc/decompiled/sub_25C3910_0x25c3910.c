// Function: sub_25C3910
// Address: 0x25c3910
//
void __fastcall sub_25C3910(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r13
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  bool v11; // zf
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v17; // rsi
  int v18; // ecx
  int v19; // r11d
  __int64 *v20; // r10
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r9
  __int64 v24; // r9
  __int64 v25; // rax
  int v26; // ecx
  __int64 *v27; // r15
  __int64 *v28; // rbx
  __int64 *v29; // r13
  __int64 v30; // rax
  int v31; // ecx
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rcx
  __int64 *k; // rcx
  __int64 *v37; // rbx
  __int64 *v38; // r8
  int v39; // esi
  int v40; // r11d
  __int64 *v41; // r10
  unsigned int v42; // ecx
  __int64 *v43; // rdi
  __int64 v44; // r9
  char v45; // al
  int v46; // esi
  __int64 *v47; // [rsp+18h] [rbp-1B8h]
  _BYTE v48[432]; // [rsp+20h] [rbp-1B0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v27 = (__int64 *)(a1 + 400);
    v47 = (__int64 *)(a1 + 16);
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
      v27 = (__int64 *)(a1 + 400);
      v47 = (__int64 *)(a1 + 16);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 96LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 6144;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 96LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v4 + v10;
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 12LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 48;
        }
        for ( i = &v13[v14]; i != v13; v13 += 12 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = v4; v12 != j; j += 96 )
        {
          v25 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v25 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = a1 + 16;
              v18 = 3;
            }
            else
            {
              v26 = *(_DWORD *)(a1 + 24);
              v17 = *(_QWORD *)(a1 + 16);
              if ( !v26 )
                goto LABEL_83;
              v18 = v26 - 1;
            }
            v19 = 1;
            v20 = 0;
            v21 = v18 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v22 = (__int64 *)(v17 + 96LL * v21);
            v23 = *v22;
            if ( *v22 != v25 )
            {
              while ( v23 != -4096 )
              {
                if ( v23 == -8192 && !v20 )
                  v20 = v22;
                v21 = v18 & (v19 + v21);
                v22 = (__int64 *)(v17 + 96LL * v21);
                v23 = *v22;
                if ( v25 == *v22 )
                  goto LABEL_16;
                ++v19;
              }
              if ( v20 )
                v22 = v20;
            }
LABEL_16:
            *v22 = v25;
            v24 = j + 16;
            *((_BYTE *)v22 + 8) = *(_BYTE *)(j + 8);
            v22[2] = (__int64)(v22 + 4);
            v22[3] = 0x200000000LL;
            if ( *(_DWORD *)(j + 24) )
            {
              sub_25C2C90((__int64)(v22 + 2), (__int64 *)(j + 16));
              v24 = j + 16;
            }
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            sub_25C0430(v24);
          }
        }
        sub_C7D6A0(v4, v10, 8);
        return;
      }
      v27 = (__int64 *)(a1 + 400);
      v2 = 64;
      v47 = (__int64 *)(a1 + 16);
    }
  }
  v28 = v47;
  v29 = (__int64 *)v48;
  do
  {
    v30 = *v28;
    if ( *v28 != -4096 && v30 != -8192 )
    {
      if ( v29 )
        *v29 = v30;
      v31 = *((_DWORD *)v28 + 6);
      v32 = (__int64)(v28 + 2);
      *((_BYTE *)v29 + 8) = *((_BYTE *)v28 + 8);
      v29[2] = (__int64)(v29 + 4);
      v29[3] = 0x200000000LL;
      if ( v31 )
      {
        sub_25C2C90((__int64)(v29 + 2), v28 + 2);
        v32 = (__int64)(v28 + 2);
      }
      v29 += 12;
      sub_25C0430(v32);
    }
    v28 += 12;
  }
  while ( v28 != v27 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(96LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(__int64 **)(a1 + 16);
    v35 = 12LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v47;
    v35 = 48;
  }
  for ( k = &v34[v35]; k != v34; v34 += 12 )
  {
    if ( v34 )
      *v34 = -4096;
  }
  v37 = (__int64 *)v48;
  if ( v29 != (__int64 *)v48 )
  {
    do
    {
      v25 = *v37;
      if ( *v37 != -8192 && v25 != -4096 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v38 = v47;
          v39 = 3;
        }
        else
        {
          v46 = *(_DWORD *)(a1 + 24);
          v38 = *(__int64 **)(a1 + 16);
          if ( !v46 )
          {
LABEL_83:
            MEMORY[0] = v25;
            BUG();
          }
          v39 = v46 - 1;
        }
        v40 = 1;
        v41 = 0;
        v42 = v39 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v43 = &v38[12 * v42];
        v44 = *v43;
        if ( v25 != *v43 )
        {
          while ( v44 != -4096 )
          {
            if ( v44 == -8192 && !v41 )
              v41 = v43;
            v42 = v39 & (v40 + v42);
            v43 = &v38[12 * v42];
            v44 = *v43;
            if ( v25 == *v43 )
              goto LABEL_51;
            ++v40;
          }
          if ( v41 )
            v43 = v41;
        }
LABEL_51:
        *v43 = v25;
        v45 = *((_BYTE *)v37 + 8);
        v43[3] = 0x200000000LL;
        *((_BYTE *)v43 + 8) = v45;
        v43[2] = (__int64)(v43 + 4);
        if ( *((_DWORD *)v37 + 6) )
          sub_25C2C90((__int64)(v43 + 2), v37 + 2);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        sub_25C0430((__int64)(v37 + 2));
      }
      v37 += 12;
    }
    while ( v29 != v37 );
  }
}
