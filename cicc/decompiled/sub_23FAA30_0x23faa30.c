// Function: sub_23FAA30
// Address: 0x23faa30
//
void __fastcall sub_23FAA30(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi

  v1 = *a1;
  if ( *((_DWORD *)a1 + 3) )
  {
    v3 = *((unsigned int *)a1 + 2);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v1 + v5);
        if ( v6 != (_QWORD *)-8LL )
        {
          if ( v6 )
          {
            sub_C7D6A0((__int64)v6, *v6 + 9LL, 8);
            v1 = *a1;
          }
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
    _libc_free(v1);
  }
  else
  {
    _libc_free(*a1);
  }
}
