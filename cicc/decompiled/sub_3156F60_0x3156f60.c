// Function: sub_3156F60
// Address: 0x3156f60
//
void __fastcall sub_3156F60(unsigned __int64 *a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi

  sub_3156F30(a1 + 7);
  sub_3156F30(a1 + 3);
  if ( *((_DWORD *)a1 + 3) )
  {
    v2 = *((unsigned int *)a1 + 2);
    v3 = *a1;
    if ( (_DWORD)v2 )
    {
      v4 = 8 * v2;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v3 + v5);
        if ( v6 != (_QWORD *)-8LL )
        {
          if ( v6 )
          {
            sub_C7D6A0((__int64)v6, *v6 + 17LL, 8);
            v3 = *a1;
          }
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
    _libc_free(v3);
  }
  else
  {
    _libc_free(*a1);
  }
}
