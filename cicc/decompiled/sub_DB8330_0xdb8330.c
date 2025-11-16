// Function: sub_DB8330
// Address: 0xdb8330
//
unsigned __int64 __fastcall sub_DB8330(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // r15
  unsigned int v11; // ebx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r13
  bool v17; // zf
  __int64 v18; // r14
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v23; // rsi
  __int64 v24; // rcx
  int v25; // r11d
  __int64 v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // r9
  __int64 v30; // rdi
  unsigned __int64 v31; // rax
  int v32; // ecx
  _BYTE *v33; // r13
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rax
  _BYTE v39[400]; // [rsp+10h] [rbp-190h] BYREF

  v6 = a2;
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( (unsigned int)a2 <= 4 )
  {
    v14 = a1 + 16;
    v15 = a1 + 368;
    if ( !(_BYTE)v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    goto LABEL_29;
  }
  v9 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  v6 = v9;
  if ( (unsigned int)v9 > 0x40 )
  {
    v14 = a1 + 16;
    v15 = a1 + 368;
    if ( !(_BYTE)v8 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_DWORD *)(a1 + 24);
      v12 = 88LL * (unsigned int)v9;
      goto LABEL_5;
    }
LABEL_29:
    v33 = v39;
    do
    {
      v34 = *(_QWORD *)v14;
      if ( *(_QWORD *)v14 != -4 && v34 != -16 )
      {
        if ( v33 )
          *(_QWORD *)v33 = v34;
        v35 = *(unsigned int *)(v14 + 48);
        *((_QWORD *)v33 + 1) = *(_QWORD *)(v14 + 8);
        *((_QWORD *)v33 + 2) = *(_QWORD *)(v14 + 16);
        *((_QWORD *)v33 + 3) = *(_QWORD *)(v14 + 24);
        v33[32] = *(_BYTE *)(v14 + 32);
        *((_QWORD *)v33 + 5) = v33 + 56;
        *((_QWORD *)v33 + 6) = 0x400000000LL;
        if ( (_DWORD)v35 )
        {
          a2 = v14 + 40;
          sub_D91460((__int64)(v33 + 40), (char **)(v14 + 40), v35, v8, a5, a6);
        }
        v36 = *(_QWORD *)(v14 + 40);
        v33 += 88;
        if ( v36 != v14 + 56 )
          _libc_free(v36, a2);
      }
      v14 += 88;
    }
    while ( v14 != v15 );
    if ( v6 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v37 = sub_C7D670(88LL * v6, 8);
      *(_DWORD *)(a1 + 24) = v6;
      *(_QWORD *)(a1 + 16) = v37;
    }
    return sub_DB8170(a1, (__int64)v39, (__int64)v33);
  }
  if ( (_BYTE)v8 )
  {
    v14 = a1 + 16;
    v15 = a1 + 368;
    v6 = 64;
    goto LABEL_29;
  }
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(_DWORD *)(a1 + 24);
  v6 = 64;
  v12 = 5632;
LABEL_5:
  v13 = sub_C7D670(v12, 8);
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 16) = v13;
LABEL_8:
  v16 = 88LL * v11;
  v17 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v18 = v10 + v16;
  if ( v17 )
  {
    v19 = *(_QWORD **)(a1 + 16);
    v20 = 11LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v19 = (_QWORD *)(a1 + 16);
    v20 = 44;
  }
  for ( i = &v19[v20]; i != v19; v19 += 11 )
  {
    if ( v19 )
      *v19 = -4;
  }
  for ( j = v10; v18 != j; j += 88 )
  {
    v31 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -16 && v31 != -4 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v23 = a1 + 16;
        v24 = 3;
      }
      else
      {
        v32 = *(_DWORD *)(a1 + 24);
        v23 = *(_QWORD *)(a1 + 16);
        if ( !v32 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v24 = (unsigned int)(v32 - 1);
      }
      v25 = 1;
      v26 = 0;
      v27 = (unsigned int)v24 & ((unsigned int)v31 ^ (unsigned int)(v31 >> 9));
      v28 = v23 + 88 * v27;
      v29 = *(_QWORD *)v28;
      if ( v31 != *(_QWORD *)v28 )
      {
        while ( v29 != -4 )
        {
          if ( !v26 && v29 == -16 )
            v26 = v28;
          a5 = (unsigned int)(v25 + 1);
          v27 = (unsigned int)v24 & (v25 + (_DWORD)v27);
          v28 = v23 + 88LL * (unsigned int)v27;
          v29 = *(_QWORD *)v28;
          if ( v31 == *(_QWORD *)v28 )
            goto LABEL_18;
          ++v25;
        }
        if ( v26 )
          v28 = v26;
      }
LABEL_18:
      *(_QWORD *)v28 = *(_QWORD *)j;
      *(_QWORD *)(v28 + 8) = *(_QWORD *)(j + 8);
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(j + 16);
      *(_QWORD *)(v28 + 24) = *(_QWORD *)(j + 24);
      *(_BYTE *)(v28 + 32) = *(_BYTE *)(j + 32);
      *(_QWORD *)(v28 + 40) = v28 + 56;
      *(_QWORD *)(v28 + 48) = 0x400000000LL;
      if ( *(_DWORD *)(j + 48) )
      {
        v23 = j + 40;
        sub_D91460(v28 + 40, (char **)(j + 40), v27, v24, a5, v29);
      }
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v30 = *(_QWORD *)(j + 40);
      if ( v30 != j + 56 )
        _libc_free(v30, v23);
    }
  }
  return sub_C7D6A0(v10, v16, 8);
}
