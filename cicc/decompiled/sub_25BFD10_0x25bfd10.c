// Function: sub_25BFD10
// Address: 0x25bfd10
//
__int64 __fastcall sub_25BFD10(unsigned __int64 *a1, __int64 *a2, int a3)
{
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( *a1 )
      {
        v5 = *(_QWORD *)(v4 + 32);
        if ( v5 != v4 + 48 )
          _libc_free(v5);
        sub_C7D6A0(*(_QWORD *)(v4 + 8), 8LL * *(unsigned int *)(v4 + 24), 8);
        j_j___libc_free_0(v4);
      }
    }
    return 0;
  }
  v6 = *a2;
  v7 = sub_22077B0(0x70u);
  v8 = v7;
  if ( v7 )
    sub_25BFBE0(v7, v6);
  *a1 = v8;
  return 0;
}
