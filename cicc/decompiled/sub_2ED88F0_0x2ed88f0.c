// Function: sub_2ED88F0
// Address: 0x2ed88f0
//
void __fastcall sub_2ED88F0(__int64 a1, unsigned int a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // ebx
  char v8; // cl
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // r15
  _DWORD *v17; // rax
  __int64 v18; // rdx
  _DWORD *v19; // r13
  __int64 v20; // r14
  __int64 v21; // r13
  _DWORD *i; // rdx
  __int64 v23; // rbx
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdx
  int v27; // r11d
  unsigned int *v28; // r10
  __int64 v29; // rcx
  unsigned int *v30; // rdi
  __int64 v31; // r9
  unsigned __int64 v32; // rdi
  __int64 v33; // rcx
  char **v34; // r15
  __int64 v35; // rax
  _DWORD *v36; // rax
  __int64 v37; // rdx
  _DWORD *j; // rdx
  char **v39; // rbx
  unsigned int v40; // eax
  _DWORD *v41; // rsi
  __int64 v42; // rcx
  int v43; // r10d
  __int64 v44; // r9
  int v45; // edx
  unsigned int *v46; // rdi
  __int64 v47; // r8
  __int64 v48; // rdx
  unsigned __int64 v49; // rdi
  int v50; // edx
  int v51; // esi
  unsigned __int64 v52; // rdi
  int v53; // edx
  __int64 v54; // [rsp+8h] [rbp-B8h]
  __int64 v55; // [rsp+8h] [rbp-B8h]
  _BYTE v56[176]; // [rsp+10h] [rbp-B0h] BYREF

  v7 = a2;
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v19 = (_DWORD *)(a1 + 16);
    v20 = a1 + 144;
    if ( !v8 )
    {
      v21 = *(unsigned int *)(a1 + 24);
      v10 = *(_QWORD *)(a1 + 16);
      *(_BYTE *)(a1 + 8) |= 1u;
      v13 = 32 * v21;
      v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v16 = v10 + v13;
      if ( v15 )
        goto LABEL_6;
      goto LABEL_9;
    }
  }
  else
  {
    a3 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 16;
    v9 = (a3
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v7 = v9;
    if ( (unsigned int)v9 > 0x40 )
    {
      v19 = (_DWORD *)(a1 + 16);
      v20 = a1 + 144;
      if ( !v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(unsigned int *)(a1 + 24);
        v12 = 32LL * (unsigned int)v9;
LABEL_5:
        v13 = 32 * v11;
        v14 = sub_C7D670(v12, 8);
        v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v16 = v10 + v13;
        *(_QWORD *)(a1 + 16) = v14;
        *(_DWORD *)(a1 + 24) = v7;
        if ( v15 )
        {
LABEL_6:
          v17 = *(_DWORD **)(a1 + 16);
          v18 = 8LL * *(unsigned int *)(a1 + 24);
LABEL_10:
          for ( i = &v17[v18]; i != v17; v17 += 8 )
          {
            if ( v17 )
              *v17 = -1;
          }
          v23 = v10;
          if ( v16 != v10 )
          {
            while ( 1 )
            {
              v24 = *(_DWORD *)v23;
              if ( *(_DWORD *)v23 > 0xFFFFFFFD )
                goto LABEL_16;
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v25 = a1 + 16;
                v26 = 3;
              }
              else
              {
                v50 = *(_DWORD *)(a1 + 24);
                v25 = *(_QWORD *)(a1 + 16);
                if ( !v50 )
                  goto LABEL_81;
                v26 = (unsigned int)(v50 - 1);
              }
              v27 = 1;
              v28 = 0;
              v29 = (unsigned int)v26 & (37 * v24);
              v30 = (unsigned int *)(v25 + 32 * v29);
              v31 = *v30;
              if ( v24 != (_DWORD)v31 )
              {
                while ( (_DWORD)v31 != -1 )
                {
                  if ( !v28 && (_DWORD)v31 == -2 )
                    v28 = v30;
                  a5 = (unsigned int)(v27 + 1);
                  v29 = (unsigned int)v26 & (v27 + (_DWORD)v29);
                  v30 = (unsigned int *)(v25 + 32LL * (unsigned int)v29);
                  v31 = *v30;
                  if ( v24 == (_DWORD)v31 )
                    goto LABEL_21;
                  ++v27;
                }
                if ( v28 )
                  v30 = v28;
              }
LABEL_21:
              *v30 = v24;
              *((_QWORD *)v30 + 1) = v30 + 6;
              *((_QWORD *)v30 + 2) = 0x200000000LL;
              if ( *(_DWORD *)(v23 + 16) )
                sub_2ED1840((__int64)(v30 + 2), (char **)(v23 + 8), v26, v29, a5, v31);
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              v32 = *(_QWORD *)(v23 + 8);
              if ( v32 == v23 + 24 )
              {
LABEL_16:
                v23 += 32;
                if ( v16 == v23 )
                  break;
              }
              else
              {
                _libc_free(v32);
                v23 += 32;
                if ( v16 == v23 )
                  break;
              }
            }
          }
          sub_C7D6A0(v10, v13, 8);
          return;
        }
LABEL_9:
        v17 = (_DWORD *)(a1 + 16);
        v18 = 32;
        goto LABEL_10;
      }
    }
    else
    {
      if ( !v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(unsigned int *)(a1 + 24);
        v7 = 64;
        v12 = 2048;
        goto LABEL_5;
      }
      v19 = (_DWORD *)(a1 + 16);
      v20 = a1 + 144;
      v7 = 64;
    }
  }
  v33 = (__int64)v19;
  v34 = (char **)v56;
  do
  {
    if ( *(_DWORD *)v33 <= 0xFFFFFFFD )
    {
      if ( v34 )
        *(_DWORD *)v34 = *(_DWORD *)v33;
      v51 = *(_DWORD *)(v33 + 16);
      v34[1] = (char *)(v34 + 3);
      v34[2] = (char *)0x200000000LL;
      if ( v51 )
      {
        v55 = v33;
        sub_2ED1840((__int64)(v34 + 1), (char **)(v33 + 8), a3, v33, a5, a6);
        v33 = v55;
      }
      v52 = *(_QWORD *)(v33 + 8);
      v34 += 4;
      if ( v52 != v33 + 24 )
      {
        v54 = v33;
        _libc_free(v52);
        v33 = v54;
      }
    }
    v33 += 32;
  }
  while ( v33 != v20 );
  if ( v7 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v35 = sub_C7D670(32LL * v7, 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v35;
  }
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v15 )
  {
    v36 = *(_DWORD **)(a1 + 16);
    v37 = 8LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v36 = v19;
    v37 = 32;
  }
  for ( j = &v36[v37]; j != v36; v36 += 8 )
  {
    if ( v36 )
      *v36 = -1;
  }
  v39 = (char **)v56;
  if ( v34 != (char **)v56 )
  {
    do
    {
      while ( 1 )
      {
        v40 = *(_DWORD *)v39;
        if ( *(_DWORD *)v39 <= 0xFFFFFFFD )
        {
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v41 = v19;
            v42 = 3;
          }
          else
          {
            v53 = *(_DWORD *)(a1 + 24);
            v41 = *(_DWORD **)(a1 + 16);
            if ( !v53 )
            {
LABEL_81:
              MEMORY[0] = 0;
              BUG();
            }
            v42 = (unsigned int)(v53 - 1);
          }
          v43 = 1;
          v44 = 0;
          v45 = v42 & (37 * v40);
          v46 = &v41[8 * v45];
          v47 = *v46;
          if ( v40 != (_DWORD)v47 )
          {
            while ( (_DWORD)v47 != -1 )
            {
              if ( (_DWORD)v47 == -2 && !v44 )
                v44 = (__int64)v46;
              v45 = v42 & (v43 + v45);
              v46 = &v41[8 * v45];
              v47 = *v46;
              if ( v40 == (_DWORD)v47 )
                goto LABEL_45;
              ++v43;
            }
            if ( v44 )
              v46 = (unsigned int *)v44;
          }
LABEL_45:
          *v46 = v40;
          *((_QWORD *)v46 + 1) = v46 + 6;
          *((_QWORD *)v46 + 2) = 0x200000000LL;
          v48 = *((unsigned int *)v39 + 4);
          if ( (_DWORD)v48 )
            sub_2ED1840((__int64)(v46 + 2), v39 + 1, v48, v42, v47, v44);
          v49 = (unsigned __int64)v39[1];
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          if ( (char **)v49 != v39 + 3 )
            break;
        }
        v39 += 4;
        if ( v34 == v39 )
          return;
      }
      _libc_free(v49);
      v39 += 4;
    }
    while ( v34 != v39 );
  }
}
