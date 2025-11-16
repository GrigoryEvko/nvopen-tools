// Function: sub_2DFFEB0
// Address: 0x2dffeb0
//
void __fastcall sub_2DFFEB0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v8; // si
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  bool v15; // zf
  __int64 v16; // r15
  _DWORD *v17; // rax
  __int64 v18; // rdx
  _DWORD *i; // rdx
  __int64 v20; // rbx
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rdx
  int v24; // r11d
  unsigned int *v25; // r10
  __int64 v26; // rcx
  unsigned int *v27; // rdi
  __int64 v28; // r9
  unsigned __int64 v29; // rdi
  int v30; // edx
  __int64 v31; // rdx
  _BYTE *v32; // rbx
  _DWORD *v33; // r14
  _BYTE *v34; // r15
  __int64 v35; // rax
  _DWORD *v36; // rax
  __int64 v37; // rcx
  _DWORD *j; // rcx
  unsigned int v39; // eax
  __int64 v40; // r9
  __int64 v41; // r8
  int v42; // r11d
  unsigned int *v43; // r10
  unsigned int v44; // esi
  unsigned int *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  unsigned __int64 v48; // rdi
  __int64 v49; // rcx
  unsigned __int64 v50; // rdi
  int v51; // ecx
  __int64 v52; // [rsp+0h] [rbp-1A0h]
  __int64 v53; // [rsp+0h] [rbp-1A0h]
  _DWORD *v54; // [rsp+8h] [rbp-198h]
  _BYTE v55[400]; // [rsp+10h] [rbp-190h] BYREF

  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v31 = a1 + 368;
    v54 = (_DWORD *)(a1 + 16);
  }
  else
  {
    v9 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    a2 = v9;
    if ( (unsigned int)v9 > 0x40 )
    {
      v31 = a1 + 368;
      v54 = (_DWORD *)(a1 + 16);
      if ( !v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(_DWORD *)(a1 + 24);
        v12 = 88LL * (unsigned int)v9;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v8 )
      {
        v10 = *(_QWORD *)(a1 + 16);
        v11 = *(_DWORD *)(a1 + 24);
        a2 = 64;
        v12 = 5632;
LABEL_5:
        v13 = sub_C7D670(v12, 8);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
        v14 = 88LL * v11;
        v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v16 = v10 + v14;
        if ( v15 )
        {
          v17 = *(_DWORD **)(a1 + 16);
          v18 = 22LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v17 = (_DWORD *)(a1 + 16);
          v18 = 88;
        }
        for ( i = &v17[v18]; i != v17; v17 += 22 )
        {
          if ( v17 )
            *v17 = -1;
        }
        v20 = v10;
        if ( v16 != v10 )
        {
          while ( 1 )
          {
            v21 = *(_DWORD *)v20;
            if ( *(_DWORD *)v20 > 0xFFFFFFFD )
              goto LABEL_16;
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v22 = a1 + 16;
              v23 = 3;
            }
            else
            {
              v30 = *(_DWORD *)(a1 + 24);
              v22 = *(_QWORD *)(a1 + 16);
              if ( !v30 )
                goto LABEL_83;
              v23 = (unsigned int)(v30 - 1);
            }
            v24 = 1;
            v25 = 0;
            v26 = (unsigned int)v23 & (37 * v21);
            v27 = (unsigned int *)(v22 + 88 * v26);
            v28 = *v27;
            if ( v21 != (_DWORD)v28 )
            {
              while ( (_DWORD)v28 != -1 )
              {
                if ( !v25 && (_DWORD)v28 == -2 )
                  v25 = v27;
                a5 = (unsigned int)(v24 + 1);
                v26 = (unsigned int)v23 & (v24 + (_DWORD)v26);
                v27 = (unsigned int *)(v22 + 88LL * (unsigned int)v26);
                v28 = *v27;
                if ( v21 == (_DWORD)v28 )
                  goto LABEL_21;
                ++v24;
              }
              if ( v25 )
                v27 = v25;
            }
LABEL_21:
            *v27 = v21;
            *((_QWORD *)v27 + 1) = v27 + 6;
            *((_QWORD *)v27 + 2) = 0x400000000LL;
            if ( *(_DWORD *)(v20 + 16) )
              sub_2DF4BD0((__int64)(v27 + 2), v20 + 8, v23, v26, a5, v28);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v29 = *(_QWORD *)(v20 + 8);
            if ( v29 == v20 + 24 )
            {
LABEL_16:
              v20 += 88;
              if ( v16 == v20 )
                break;
            }
            else
            {
              _libc_free(v29);
              v20 += 88;
              if ( v16 == v20 )
                break;
            }
          }
        }
        sub_C7D6A0(v10, v14, 8);
        return;
      }
      v31 = a1 + 368;
      a2 = 64;
      v54 = (_DWORD *)(a1 + 16);
    }
  }
  v32 = v55;
  v33 = v54;
  v34 = v55;
  do
  {
    if ( *v33 <= 0xFFFFFFFD )
    {
      if ( v34 )
        *(_DWORD *)v34 = *v33;
      v49 = (unsigned int)v33[4];
      *((_QWORD *)v34 + 1) = v34 + 24;
      *((_QWORD *)v34 + 2) = 0x400000000LL;
      if ( (_DWORD)v49 )
      {
        v53 = v31;
        sub_2DF4BD0((__int64)(v34 + 8), (__int64)(v33 + 2), v31, v49, a5, a6);
        v31 = v53;
      }
      v50 = *((_QWORD *)v33 + 1);
      v34 += 88;
      if ( (_DWORD *)v50 != v33 + 6 )
      {
        v52 = v31;
        _libc_free(v50);
        v31 = v52;
      }
    }
    v33 += 22;
  }
  while ( v33 != (_DWORD *)v31 );
  if ( a2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v35 = sub_C7D670(88LL * a2, 8);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v35;
  }
  v15 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v15 )
  {
    v36 = *(_DWORD **)(a1 + 16);
    v37 = 22LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v36 = v54;
    v37 = 88;
  }
  for ( j = &v36[v37]; j != v36; v36 += 22 )
  {
    if ( v36 )
      *v36 = -1;
  }
  if ( v34 != v55 )
  {
    do
    {
      while ( 1 )
      {
        v39 = *(_DWORD *)v32;
        if ( *(_DWORD *)v32 <= 0xFFFFFFFD )
        {
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v40 = (__int64)v54;
            v41 = 3;
          }
          else
          {
            v51 = *(_DWORD *)(a1 + 24);
            v40 = *(_QWORD *)(a1 + 16);
            if ( !v51 )
            {
LABEL_83:
              MEMORY[0] = 0;
              BUG();
            }
            v41 = (unsigned int)(v51 - 1);
          }
          v42 = 1;
          v43 = 0;
          v44 = v41 & (37 * v39);
          v45 = (unsigned int *)(v40 + 88LL * v44);
          v46 = *v45;
          if ( v39 != (_DWORD)v46 )
          {
            while ( (_DWORD)v46 != -1 )
            {
              if ( (_DWORD)v46 == -2 && !v43 )
                v43 = v45;
              v44 = v41 & (v42 + v44);
              v45 = (unsigned int *)(v40 + 88LL * v44);
              v46 = *v45;
              if ( v39 == (_DWORD)v46 )
                goto LABEL_48;
              ++v42;
            }
            if ( v43 )
              v45 = v43;
          }
LABEL_48:
          *v45 = v39;
          *((_QWORD *)v45 + 1) = v45 + 6;
          *((_QWORD *)v45 + 2) = 0x400000000LL;
          v47 = *((unsigned int *)v32 + 4);
          if ( (_DWORD)v47 )
            sub_2DF4BD0((__int64)(v45 + 2), (__int64)(v32 + 8), v47, v46, v41, v40);
          v48 = *((_QWORD *)v32 + 1);
          *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          if ( (_BYTE *)v48 != v32 + 24 )
            break;
        }
        v32 += 88;
        if ( v34 == v32 )
          return;
      }
      _libc_free(v48);
      v32 += 88;
    }
    while ( v34 != v32 );
  }
}
