// Function: sub_1B2C350
// Address: 0x1b2c350
//
__int64 __fastcall sub_1B2C350(unsigned __int64 **a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  unsigned __int64 *v11; // r12
  __int64 v12; // rbx
  unsigned __int64 *v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rdx
  char **v16; // rsi
  __int64 v17; // rdi
  unsigned __int64 *v18; // rbx
  unsigned __int64 v19; // rdi
  __int64 v21; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v4 = ((((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
      | (*((unsigned int *)a1 + 3) + 2LL)
      | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
       | (*((unsigned int *)a1 + 3) + 2LL)
       | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 8)
     | v4
     | (((*((unsigned int *)a1 + 3) + 2LL) | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1)) >> 2)
     | (*((unsigned int *)a1 + 3) + 2LL)
     | (((unsigned __int64)*((unsigned int *)a1 + 3) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = v3;
  if ( v3 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v21 = malloc(96 * v7);
  if ( !v21 )
    sub_16BD1C0("Allocation failed", 1u);
  v11 = &(*a1)[12 * *((unsigned int *)a1 + 2)];
  if ( *a1 != v11 )
  {
    v12 = v21;
    v13 = *a1;
    do
    {
      while ( 1 )
      {
        if ( v12 )
        {
          *(_DWORD *)(v12 + 8) = 0;
          *(_QWORD *)v12 = v12 + 16;
          *(_DWORD *)(v12 + 12) = 4;
          v15 = *((unsigned int *)v13 + 2);
          if ( (_DWORD)v15 )
            sub_1B29D10(v12, (char **)v13, v15, v8, v9, v10);
          v14 = v12 + 64;
          *(_DWORD *)(v12 + 56) = 0;
          *(_QWORD *)(v12 + 48) = v12 + 64;
          *(_DWORD *)(v12 + 60) = 4;
          if ( *((_DWORD *)v13 + 14) )
            break;
        }
        v13 += 12;
        v12 += 96;
        if ( v11 == v13 )
          goto LABEL_17;
      }
      v16 = (char **)(v13 + 6);
      v17 = v12 + 48;
      v13 += 12;
      v12 += 96;
      sub_1B29D10(v17, v16, v14, v8, v9, v10);
    }
    while ( v11 != v13 );
LABEL_17:
    v18 = *a1;
    v11 = &(*a1)[12 * *((unsigned int *)a1 + 2)];
    if ( *a1 != v11 )
    {
      do
      {
        v11 -= 12;
        v19 = v11[6];
        if ( (unsigned __int64 *)v19 != v11 + 8 )
          _libc_free(v19);
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          _libc_free(*v11);
      }
      while ( v11 != v18 );
      v11 = *a1;
    }
  }
  if ( v11 != (unsigned __int64 *)(a1 + 2) )
    _libc_free((unsigned __int64)v11);
  *((_DWORD *)a1 + 3) = v7;
  *a1 = (unsigned __int64 *)v21;
  return v21;
}
