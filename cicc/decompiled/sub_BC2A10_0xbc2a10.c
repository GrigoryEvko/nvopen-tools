// Function: sub_BC2A10
// Address: 0xbc2a10
//
void __fastcall sub_BC2A10(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *v4; // r13
  _QWORD *v5; // r12
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // r12
  _QWORD *v10; // rdi

  v3 = a1[18];
  if ( v3 )
  {
    a2 = a1[20] - v3;
    j_j___libc_free_0(v3, a2);
  }
  v4 = (_QWORD *)a1[16];
  v5 = (_QWORD *)a1[15];
  if ( v4 != v5 )
  {
    do
    {
      if ( *v5 )
      {
        a2 = 56;
        j_j___libc_free_0(*v5, 56);
      }
      ++v5;
    }
    while ( v4 != v5 );
    v5 = (_QWORD *)a1[15];
  }
  if ( v5 )
  {
    a2 = a1[17] - (_QWORD)v5;
    j_j___libc_free_0(v5, a2);
  }
  v6 = a1[12];
  if ( *((_DWORD *)a1 + 27) )
  {
    v7 = *((unsigned int *)a1 + 26);
    if ( (_DWORD)v7 )
    {
      v8 = 8 * v7;
      v9 = 0;
      do
      {
        v10 = *(_QWORD **)(v6 + v9);
        if ( v10 && v10 != (_QWORD *)-8LL )
        {
          a2 = *v10 + 17LL;
          sub_C7D6A0(v10, a2, 8);
          v6 = a1[12];
        }
        v9 += 8;
      }
      while ( v8 != v9 );
    }
  }
  _libc_free(v6, a2);
  sub_C7D6A0(a1[9], 16LL * *((unsigned int *)a1 + 22), 8);
}
