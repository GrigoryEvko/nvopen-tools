// Function: sub_2E6DCE0
// Address: 0x2e6dce0
//
void __fastcall sub_2E6DCE0(__int64 *a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi

  v2 = *a1;
  v3 = *a1 + 8LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8;
      if ( v4 )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( v5 != v4 + 40 )
          _libc_free(v5);
        j_j___libc_free_0(v4);
      }
    }
    while ( v3 != v2 );
  }
  *((_DWORD *)a1 + 2) = 0;
}
