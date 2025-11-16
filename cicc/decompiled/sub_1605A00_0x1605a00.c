// Function: sub_1605A00
// Address: 0x1605a00
//
void __fastcall sub_1605A00(__int64 a1)
{
  unsigned __int64 v1; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi

  v1 = *(_QWORD *)a1;
  if ( *(_DWORD *)(a1 + 12) )
  {
    v3 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD *)(v1 + v5);
        if ( v6 )
        {
          if ( v6 != -8 )
          {
            _libc_free(v6);
            v1 = *(_QWORD *)a1;
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
    _libc_free(*(_QWORD *)a1);
  }
}
