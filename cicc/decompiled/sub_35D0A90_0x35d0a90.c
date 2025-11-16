// Function: sub_35D0A90
// Address: 0x35d0a90
//
__int64 __fastcall sub_35D0A90(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r13d
  char v8; // dl
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // r12
  _DWORD *v16; // rax
  __int64 v17; // rdx
  _DWORD *i; // rdx
  __int64 v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // r13
  __int64 v22; // r9
  __int64 v23; // r8
  int v24; // r10d
  unsigned int v25; // edx
  unsigned int *v26; // rsi
  unsigned int *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r13
  unsigned __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rdi
  int v37; // edx
  __int64 v38; // rbx
  unsigned int *v39; // r12
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rdi
  __int64 v45; // [rsp+8h] [rbp-118h]
  __int64 v46; // [rsp+8h] [rbp-118h]
  unsigned int v47[68]; // [rsp+10h] [rbp-110h] BYREF

  v7 = a2;
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
    v38 = a1 + 16;
    v46 = a1 + 240;
    goto LABEL_31;
  }
  a4 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
      | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
      | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
      | (a2 - 1)
      | ((unsigned __int64)(a2 - 1) >> 1)) >> 16;
  v9 = (a4
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
    a4 = a1 + 240;
    v38 = a1 + 16;
    v46 = a1 + 240;
    if ( !v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      v12 = 56LL * (unsigned int)v9;
      goto LABEL_5;
    }
LABEL_31:
    v39 = v47;
    do
    {
      v40 = v38 + 56;
      if ( (unsigned int)(*(_DWORD *)v38 + 0x7FFFFFFF) <= 0xFFFFFFFD )
      {
        if ( v39 )
          *v39 = *(_DWORD *)v38;
        v42 = *(_QWORD *)(v38 + 16);
        v43 = *(unsigned int *)(v38 + 48);
        *((_QWORD *)v39 + 1) = 1;
        ++*(_QWORD *)(v38 + 8);
        *((_QWORD *)v39 + 2) = v42;
        LODWORD(v42) = *(_DWORD *)(v38 + 24);
        *(_QWORD *)(v38 + 16) = 0;
        v39[6] = v42;
        LODWORD(v42) = *(_DWORD *)(v38 + 28);
        *(_DWORD *)(v38 + 24) = 0;
        v39[7] = v42;
        LODWORD(v42) = *(_DWORD *)(v38 + 32);
        *(_DWORD *)(v38 + 28) = 0;
        v39[8] = v42;
        *(_DWORD *)(v38 + 32) = 0;
        *((_QWORD *)v39 + 5) = v39 + 14;
        *((_QWORD *)v39 + 6) = 0;
        if ( (_DWORD)v43 )
          sub_35CFBA0((__int64)(v39 + 10), (char **)(v38 + 40), v43, a4, a5, a6);
        v44 = *(_QWORD *)(v38 + 40);
        v39 += 14;
        v40 = v38 + 56;
        if ( v44 != v38 + 56 )
          _libc_free(v44);
        sub_C7D6A0(*(_QWORD *)(v38 + 16), 8LL * *(unsigned int *)(v38 + 32), 8);
      }
      v38 = v40;
    }
    while ( v40 != v46 );
    if ( v7 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v41 = sub_C7D670(56LL * v7, 8);
      *(_DWORD *)(a1 + 24) = v7;
      *(_QWORD *)(a1 + 16) = v41;
    }
    return sub_35D0870(a1, v47, v39);
  }
  if ( v8 )
  {
    v38 = a1 + 16;
    v7 = 64;
    v46 = a1 + 240;
    goto LABEL_31;
  }
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(_DWORD *)(a1 + 24);
  v7 = 64;
  v12 = 3584;
LABEL_5:
  v13 = sub_C7D670(v12, 8);
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v45 = 56LL * v11;
  v15 = v10 + v45;
  if ( v14 )
  {
    v16 = *(_DWORD **)(a1 + 16);
    v17 = 14LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v16 = (_DWORD *)(a1 + 16);
    v17 = 56;
  }
  for ( i = &v16[v17]; i != v16; v16 += 14 )
  {
    if ( v16 )
      *v16 = 0x7FFFFFFF;
  }
  v19 = v10;
  if ( v15 != v10 )
  {
    do
    {
      while ( 1 )
      {
        v20 = *(_DWORD *)v19;
        v21 = v19 + 56;
        if ( (unsigned int)(*(_DWORD *)v19 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v19 += 56;
        if ( v15 == v21 )
          return sub_C7D6A0(v10, v45, 8);
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v22 = a1 + 16;
        v23 = 3;
      }
      else
      {
        v37 = *(_DWORD *)(a1 + 24);
        v22 = *(_QWORD *)(a1 + 16);
        if ( !v37 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v23 = (unsigned int)(v37 - 1);
      }
      v24 = 1;
      v25 = v23 & (37 * v20);
      v26 = 0;
      v27 = (unsigned int *)(v22 + 56LL * v25);
      v28 = *v27;
      if ( v20 != (_DWORD)v28 )
      {
        while ( (_DWORD)v28 != 0x7FFFFFFF )
        {
          if ( !v26 && (_DWORD)v28 == 0x80000000 )
            v26 = v27;
          v25 = v23 & (v24 + v25);
          v27 = (unsigned int *)(v22 + 56LL * v25);
          v28 = *v27;
          if ( v20 == (_DWORD)v28 )
            goto LABEL_21;
          ++v24;
        }
        if ( v26 )
          v27 = v26;
      }
LABEL_21:
      *((_QWORD *)v27 + 3) = 0;
      *((_QWORD *)v27 + 2) = 0;
      v27[8] = 0;
      *v27 = v20;
      *((_QWORD *)v27 + 1) = 1;
      v29 = *(_QWORD *)(v19 + 16);
      ++*(_QWORD *)(v19 + 8);
      v30 = *((_QWORD *)v27 + 2);
      *((_QWORD *)v27 + 2) = v29;
      LODWORD(v29) = *(_DWORD *)(v19 + 24);
      *(_QWORD *)(v19 + 16) = v30;
      LODWORD(v30) = v27[6];
      v27[6] = v29;
      LODWORD(v29) = *(_DWORD *)(v19 + 28);
      *(_DWORD *)(v19 + 24) = v30;
      LODWORD(v30) = v27[7];
      v27[7] = v29;
      v31 = *(unsigned int *)(v19 + 32);
      *(_DWORD *)(v19 + 28) = v30;
      LODWORD(v30) = v27[8];
      v27[8] = v31;
      *(_DWORD *)(v19 + 32) = v30;
      *((_QWORD *)v27 + 5) = v27 + 14;
      *((_QWORD *)v27 + 6) = 0;
      if ( *(_DWORD *)(v19 + 48) )
        sub_35CFBA0((__int64)(v27 + 10), (char **)(v19 + 40), v31, v28, v23, v22);
      v32 = v19 + 56;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v33 = *(_QWORD *)(v19 + 40);
      if ( v33 != v19 + 56 )
        _libc_free(v33);
      v34 = *(unsigned int *)(v19 + 32);
      v35 = *(_QWORD *)(v19 + 16);
      v19 += 56;
      sub_C7D6A0(v35, 8 * v34, 8);
    }
    while ( v15 != v32 );
  }
  return sub_C7D6A0(v10, v45, 8);
}
