// Function: sub_1F53020
// Address: 0x1f53020
//
void __fastcall sub_1F53020(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  int v8; // r13d
  __int64 v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rbx
  __int64 v12; // r9
  _BYTE *v13; // r14
  _DWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rcx
  _DWORD *v17; // rcx
  _DWORD *v18; // r8
  int v19; // eax
  _DWORD *v20; // r9
  int v21; // r8d
  int v22; // r11d
  int *v23; // r10
  __int64 v24; // rcx
  int *v25; // rdi
  int v26; // esi
  __int64 v27; // rdx
  unsigned __int64 v28; // rdi
  int *v29; // r14
  unsigned int v30; // ebx
  bool v31; // zf
  int *v32; // r15
  _DWORD *v33; // rax
  __int64 v34; // rdx
  _DWORD *i; // rdx
  int *j; // rbx
  unsigned int v37; // eax
  __int64 v38; // r9
  int v39; // r8d
  int v40; // r10d
  __int64 v41; // rdx
  unsigned int *v42; // rsi
  unsigned int *v43; // rdi
  __int64 v44; // rcx
  unsigned __int64 v45; // rdi
  int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rcx
  unsigned __int64 v49; // rdi
  int v50; // ecx
  __int64 v51; // [rsp+8h] [rbp-128h]
  __int64 v52; // [rsp+8h] [rbp-128h]
  __int64 v53; // [rsp+10h] [rbp-120h]
  __int64 v54; // [rsp+10h] [rbp-120h]
  _DWORD *v55; // [rsp+18h] [rbp-118h]
  _BYTE v56[272]; // [rsp+20h] [rbp-110h] BYREF

  v6 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( (_BYTE)v6 )
      return;
    v29 = *(int **)(a1 + 16);
    v30 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
  }
  else
  {
    v7 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v8 = v7;
    if ( (unsigned int)v7 > 0x40 )
    {
      a5 = 7 * v7;
      v9 = 14LL * (unsigned int)v7;
      if ( (_BYTE)v6 )
        goto LABEL_5;
      v29 = *(int **)(a1 + 16);
      v30 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( (_BYTE)v6 )
      {
        v9 = 896;
        v8 = 64;
LABEL_5:
        v10 = a1 + 16;
        v11 = v56;
        v55 = (_DWORD *)(a1 + 16);
        v12 = a1 + 240;
        v13 = v56;
        do
        {
          if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
          {
            if ( v13 )
              *(_DWORD *)v13 = *(_DWORD *)v10;
            *((_QWORD *)v13 + 1) = v13 + 24;
            v48 = *(unsigned int *)(v10 + 16);
            *((_QWORD *)v13 + 2) = 0x400000000LL;
            if ( (_DWORD)v48 )
            {
              v52 = v12;
              v54 = v10;
              sub_1F4C4E0((__int64)(v13 + 8), v10 + 8, v6, v48, a5, v12);
              v12 = v52;
              v10 = v54;
            }
            v49 = *(_QWORD *)(v10 + 8);
            v13 += 56;
            if ( v49 != v10 + 24 )
            {
              v51 = v12;
              v53 = v10;
              _libc_free(v49);
              v12 = v51;
              v10 = v53;
            }
          }
          v10 += 56;
        }
        while ( v10 != v12 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v14 = (_DWORD *)sub_22077B0(v9 * 4);
        v15 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v8;
        *(_QWORD *)(a1 + 16) = v14;
        v16 = v15 & 1;
        if ( (_BYTE)v16 )
          v14 = v55;
        *(_QWORD *)(a1 + 8) = v16;
        if ( (_BYTE)v16 )
          v9 = 56;
        v17 = v14;
        v18 = &v14[v9];
        while ( 1 )
        {
          if ( v17 )
            *v14 = -1;
          v14 += 14;
          if ( v18 == v14 )
            break;
          v17 = v14;
        }
        if ( v13 != v56 )
        {
          while ( 1 )
          {
            v19 = *(_DWORD *)v11;
            if ( *(_DWORD *)v11 <= 0xFFFFFFFD )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v20 = v55;
                v21 = 3;
              }
              else
              {
                v50 = *(_DWORD *)(a1 + 24);
                v20 = *(_DWORD **)(a1 + 16);
                if ( !v50 )
                  goto LABEL_82;
                v21 = v50 - 1;
              }
              v22 = 1;
              v23 = 0;
              v24 = v21 & (unsigned int)(37 * v19);
              v25 = &v20[14 * v24];
              v26 = *v25;
              if ( v19 != *v25 )
              {
                while ( v26 != -1 )
                {
                  if ( v26 == -2 && !v23 )
                    v23 = v25;
                  v24 = v21 & (unsigned int)(v22 + v24);
                  v25 = &v20[14 * (unsigned int)v24];
                  v26 = *v25;
                  if ( v19 == *v25 )
                    goto LABEL_24;
                  ++v22;
                }
                if ( v23 )
                  v25 = v23;
              }
LABEL_24:
              *v25 = v19;
              *((_QWORD *)v25 + 1) = v25 + 6;
              *((_QWORD *)v25 + 2) = 0x400000000LL;
              v27 = *((unsigned int *)v11 + 4);
              if ( (_DWORD)v27 )
                sub_1F4C4E0((__int64)(v25 + 2), (__int64)(v11 + 8), v27, v24, v21, (int)v20);
              v28 = *((_QWORD *)v11 + 1);
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              if ( (_BYTE *)v28 != v11 + 24 )
                _libc_free(v28);
            }
            v11 += 56;
            if ( v13 == v11 )
              return;
          }
        }
        return;
      }
      v29 = *(int **)(a1 + 16);
      v30 = *(_DWORD *)(a1 + 24);
      v9 = 896;
      v8 = 64;
    }
    v47 = sub_22077B0(v9 * 4);
    *(_DWORD *)(a1 + 24) = v8;
    *(_QWORD *)(a1 + 16) = v47;
  }
  v31 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v32 = &v29[14 * v30];
  if ( v31 )
  {
    v33 = *(_DWORD **)(a1 + 16);
    v34 = 14LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v33 = (_DWORD *)(a1 + 16);
    v34 = 56;
  }
  for ( i = &v33[v34]; i != v33; v33 += 14 )
  {
    if ( v33 )
      *v33 = -1;
  }
  for ( j = v29; v32 != j; j += 14 )
  {
    while ( 1 )
    {
      v37 = *j;
      if ( (unsigned int)*j <= 0xFFFFFFFD )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v38 = a1 + 16;
          v39 = 3;
        }
        else
        {
          v46 = *(_DWORD *)(a1 + 24);
          v38 = *(_QWORD *)(a1 + 16);
          if ( !v46 )
          {
LABEL_82:
            MEMORY[0] = 0;
            BUG();
          }
          v39 = v46 - 1;
        }
        v40 = 1;
        v41 = v39 & (37 * v37);
        v42 = 0;
        v43 = (unsigned int *)(v38 + 56 * v41);
        v44 = *v43;
        if ( v37 != (_DWORD)v44 )
        {
          while ( (_DWORD)v44 != -1 )
          {
            if ( (_DWORD)v44 == -2 && !v42 )
              v42 = v43;
            v41 = v39 & (unsigned int)(v40 + v41);
            v43 = (unsigned int *)(v38 + 56LL * (unsigned int)v41);
            v44 = *v43;
            if ( v37 == (_DWORD)v44 )
              goto LABEL_44;
            ++v40;
          }
          if ( v42 )
            v43 = v42;
        }
LABEL_44:
        *v43 = v37;
        *((_QWORD *)v43 + 1) = v43 + 6;
        *((_QWORD *)v43 + 2) = 0x400000000LL;
        if ( j[4] )
          sub_1F4C4E0((__int64)(v43 + 2), (__int64)(j + 2), v41, v44, v39, v38);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v45 = *((_QWORD *)j + 1);
        if ( (int *)v45 != j + 6 )
          break;
      }
      j += 14;
      if ( v32 == j )
        goto LABEL_48;
    }
    _libc_free(v45);
  }
LABEL_48:
  j___libc_free_0(v29);
}
