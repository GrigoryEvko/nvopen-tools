// Function: sub_1984210
// Address: 0x1984210
//
__int64 __fastcall sub_1984210(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  __int64 **v11; // r12
  __int64 v12; // rbx
  char **v13; // r15
  char *v14; // rdx
  __int64 **v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  __int64 v19; // [rsp+8h] [rbp-38h]

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
  v19 = malloc(320 * v7);
  if ( !v19 )
    sub_16BD1C0("Allocation failed", 1u);
  v11 = (__int64 **)(*a1 + 320LL * *((unsigned int *)a1 + 2));
  if ( (__int64 **)*a1 != v11 )
  {
    v12 = v19;
    v13 = (char **)*a1;
    do
    {
      if ( v12 )
      {
        v14 = *v13;
        *(_DWORD *)(v12 + 16) = 0;
        *(_DWORD *)(v12 + 20) = 16;
        *(_QWORD *)v12 = v14;
        *(_QWORD *)(v12 + 8) = v12 + 24;
        if ( *((_DWORD *)v13 + 4) )
          sub_1983B00(v12 + 8, v13 + 1, v12 + 24, v8, v9, v10);
        sub_16CCEE0((_QWORD *)(v12 + 152), v12 + 192, 16, (__int64)(v13 + 19));
      }
      v13 += 40;
      v12 += 320;
    }
    while ( v11 != (__int64 **)v13 );
    v15 = (__int64 **)*a1;
    v11 = (__int64 **)(*a1 + 320LL * *((unsigned int *)a1 + 2));
    if ( (__int64 **)*a1 != v11 )
    {
      do
      {
        v11 -= 40;
        v16 = (unsigned __int64)v11[21];
        if ( (__int64 *)v16 != v11[20] )
          _libc_free(v16);
        v17 = (unsigned __int64)v11[1];
        if ( (__int64 **)v17 != v11 + 3 )
          _libc_free(v17);
      }
      while ( v11 != v15 );
      v11 = (__int64 **)*a1;
    }
  }
  if ( v11 != (__int64 **)(a1 + 2) )
    _libc_free((unsigned __int64)v11);
  *((_DWORD *)a1 + 3) = v7;
  *a1 = v19;
  return v19;
}
