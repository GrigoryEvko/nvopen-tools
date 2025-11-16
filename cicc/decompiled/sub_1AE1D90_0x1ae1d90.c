// Function: sub_1AE1D90
// Address: 0x1ae1d90
//
void __fastcall sub_1AE1D90(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rdx
  __int64 v7; // r14
  unsigned __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // rbx
  __int64 v11; // r9
  __int64 v12; // rcx
  char **v13; // r15
  char *v14; // rax
  int v15; // esi
  unsigned __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rcx
  _QWORD *v20; // rcx
  _QWORD *v21; // r8
  char **v22; // rbx
  __int64 v23; // rsi
  int v24; // r8d
  int v25; // r11d
  char **v26; // r10
  __int64 v27; // rcx
  char **v28; // rdi
  char *v29; // r9
  __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  char *v32; // rax
  int v33; // ecx
  __int64 v34; // r15
  __int64 v35; // r15
  bool v36; // zf
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v41; // rcx
  int v42; // esi
  int v43; // r10d
  char **v44; // r9
  __int64 v45; // rdx
  char **v46; // rdi
  char *v47; // r8
  unsigned __int64 v48; // rdi
  int v49; // edx
  __int64 v50; // rax
  __int64 v51; // [rsp+10h] [rbp-C0h]
  __int64 v52; // [rsp+10h] [rbp-C0h]
  __int64 v53; // [rsp+18h] [rbp-B8h]
  __int64 v54; // [rsp+18h] [rbp-B8h]
  _BYTE v55[176]; // [rsp+20h] [rbp-B0h] BYREF

  v6 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( (_BYTE)v6 )
      return;
    v7 = *(_QWORD *)(a1 + 16);
    v34 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v9 = v8;
    if ( (unsigned int)v8 > 0x40 )
    {
      a5 = v8;
      v10 = 4LL * (unsigned int)v8;
      if ( (_BYTE)v6 )
        goto LABEL_5;
      v34 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( (_BYTE)v6 )
      {
        v10 = 256;
        v9 = 64;
LABEL_5:
        v11 = a1 + 144;
        v12 = a1 + 16;
        v13 = (char **)v55;
        do
        {
          v14 = *(char **)v12;
          if ( *(_QWORD *)v12 != -8 && v14 != (char *)-16LL )
          {
            if ( v13 )
              *v13 = v14;
            v15 = *(_DWORD *)(v12 + 16);
            v13[1] = (char *)(v13 + 3);
            v13[2] = (char *)0x100000000LL;
            if ( v15 )
            {
              v52 = v11;
              v54 = v12;
              sub_1AE1780((__int64)(v13 + 1), (char **)(v12 + 8), v6, v12, a5, v11);
              v11 = v52;
              v12 = v54;
            }
            v16 = *(_QWORD *)(v12 + 8);
            v13 += 4;
            if ( v16 != v12 + 24 )
            {
              v51 = v11;
              v53 = v12;
              _libc_free(v16);
              v11 = v51;
              v12 = v53;
            }
          }
          v12 += 32;
        }
        while ( v12 != v11 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v17 = (_QWORD *)sub_22077B0(v10 * 8);
        v18 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v9;
        *(_QWORD *)(a1 + 16) = v17;
        v19 = v18 & 1;
        *(_QWORD *)(a1 + 8) = v19;
        if ( (_BYTE)v19 )
        {
          v17 = (_QWORD *)(a1 + 16);
          v10 = 16;
        }
        v20 = v17;
        v21 = &v17[v10];
        while ( 1 )
        {
          if ( v20 )
            *v17 = -8;
          v17 += 4;
          if ( v21 == v17 )
            break;
          v20 = v17;
        }
        v22 = (char **)v55;
        if ( v13 != (char **)v55 )
        {
          while ( 1 )
          {
            v32 = *v22;
            if ( *v22 != (char *)-8LL && v32 != (char *)-16LL )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v23 = a1 + 16;
                v24 = 3;
              }
              else
              {
                v33 = *(_DWORD *)(a1 + 24);
                v23 = *(_QWORD *)(a1 + 16);
                if ( !v33 )
                  goto LABEL_83;
                v24 = v33 - 1;
              }
              v25 = 1;
              v26 = 0;
              v27 = v24 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
              v28 = (char **)(v23 + 32 * v27);
              v29 = *v28;
              if ( v32 != *v28 )
              {
                while ( v29 != (char *)-8LL )
                {
                  if ( !v26 && v29 == (char *)-16LL )
                    v26 = v28;
                  v27 = v24 & (unsigned int)(v25 + v27);
                  v28 = (char **)(v23 + 32LL * (unsigned int)v27);
                  v29 = *v28;
                  if ( v32 == *v28 )
                    goto LABEL_26;
                  ++v25;
                }
                if ( v26 )
                  v28 = v26;
              }
LABEL_26:
              *v28 = v32;
              v28[1] = (char *)(v28 + 3);
              v28[2] = (char *)0x100000000LL;
              v30 = *((unsigned int *)v22 + 4);
              if ( (_DWORD)v30 )
                sub_1AE1780((__int64)(v28 + 1), v22 + 1, v30, v27, v24, (int)v29);
              v31 = (unsigned __int64)v22[1];
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              if ( (char **)v31 != v22 + 3 )
                _libc_free(v31);
            }
            v22 += 4;
            if ( v13 == v22 )
              return;
          }
        }
        return;
      }
      v34 = *(unsigned int *)(a1 + 24);
      v10 = 256;
      v9 = 64;
    }
    v50 = sub_22077B0(v10 * 8);
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v50;
  }
  v35 = v7 + 32 * v34;
  v36 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v36 )
  {
    v37 = *(_QWORD **)(a1 + 16);
    v38 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v37 = (_QWORD *)(a1 + 16);
    v38 = 16;
  }
  for ( i = &v37[v38]; i != v37; v37 += 4 )
  {
    if ( v37 )
      *v37 = -8;
  }
  for ( j = v7; v35 != j; j += 32 )
  {
    v32 = *(char **)j;
    if ( *(_QWORD *)j != -16 && v32 != (char *)-8LL )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v41 = a1 + 16;
        v42 = 3;
      }
      else
      {
        v49 = *(_DWORD *)(a1 + 24);
        v41 = *(_QWORD *)(a1 + 16);
        if ( !v49 )
        {
LABEL_83:
          MEMORY[0] = v32;
          BUG();
        }
        v42 = v49 - 1;
      }
      v43 = 1;
      v44 = 0;
      v45 = v42 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v46 = (char **)(v41 + 32 * v45);
      v47 = *v46;
      if ( *v46 != v32 )
      {
        while ( v47 != (char *)-8LL )
        {
          if ( !v44 && v47 == (char *)-16LL )
            v44 = v46;
          v45 = v42 & (unsigned int)(v43 + v45);
          v46 = (char **)(v41 + 32LL * (unsigned int)v45);
          v47 = *v46;
          if ( v32 == *v46 )
            goto LABEL_49;
          ++v43;
        }
        if ( v44 )
          v46 = v44;
      }
LABEL_49:
      *v46 = v32;
      v46[1] = (char *)(v46 + 3);
      v46[2] = (char *)0x100000000LL;
      if ( *(_DWORD *)(j + 16) )
        sub_1AE1780((__int64)(v46 + 1), (char **)(j + 8), v45, v41, (int)v47, (int)v44);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v48 = *(_QWORD *)(j + 8);
      if ( v48 != j + 24 )
        _libc_free(v48);
    }
  }
  j___libc_free_0(v7);
}
