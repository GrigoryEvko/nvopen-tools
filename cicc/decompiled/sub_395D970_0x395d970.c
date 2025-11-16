// Function: sub_395D970
// Address: 0x395d970
//
__int64 __fastcall sub_395D970(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  int v14; // edx
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v20; // [rsp+8h] [rbp-38h]

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
  v20 = malloc(120 * v7);
  if ( !v20 )
    sub_16BD1C0("Allocation failed", 1u);
  v10 = *((unsigned int *)a1 + 2);
  v11 = *a1 + 120 * v10;
  if ( *a1 != v11 )
  {
    v12 = v20;
    v13 = *a1;
    do
    {
      if ( v12 )
      {
        *(_DWORD *)v12 = *(_DWORD *)v13;
        v14 = *(_DWORD *)(v13 + 4);
        *(_DWORD *)(v12 + 16) = 0;
        *(_DWORD *)(v12 + 4) = v14;
        *(_QWORD *)(v12 + 8) = v12 + 24;
        *(_DWORD *)(v12 + 20) = 4;
        v15 = *(unsigned int *)(v13 + 16);
        if ( (_DWORD)v15 )
          sub_3959170(v12 + 8, (char **)(v13 + 8), v15, v10, v8, v9);
        *(_DWORD *)(v12 + 64) = 0;
        *(_QWORD *)(v12 + 56) = v12 + 72;
        *(_DWORD *)(v12 + 68) = 4;
        if ( *(_DWORD *)(v13 + 64) )
          sub_3959170(v12 + 56, (char **)(v13 + 56), v12 + 72, v10, v8, v9);
        *(_QWORD *)(v12 + 104) = *(_QWORD *)(v13 + 104);
        *(_DWORD *)(v12 + 112) = *(_DWORD *)(v13 + 112);
      }
      v13 += 120LL;
      v12 += 120;
    }
    while ( v11 != v13 );
    v16 = *a1;
    v11 = *a1 + 120LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v11 )
    {
      do
      {
        v11 -= 120LL;
        v17 = *(_QWORD *)(v11 + 56);
        if ( v17 != v11 + 72 )
          _libc_free(v17);
        v18 = *(_QWORD *)(v11 + 8);
        if ( v18 != v11 + 24 )
          _libc_free(v18);
      }
      while ( v11 != v16 );
      v11 = *a1;
    }
  }
  if ( (unsigned __int64 *)v11 != a1 + 2 )
    _libc_free(v11);
  *((_DWORD *)a1 + 3) = v7;
  *a1 = v20;
  return v20;
}
