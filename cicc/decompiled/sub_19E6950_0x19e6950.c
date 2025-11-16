// Function: sub_19E6950
// Address: 0x19e6950
//
void __fastcall sub_19E6950(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = &v2[8 * v1];
    do
    {
      if ( *v2 != 0x7FFFFFFF0LL && *v2 != -8 )
      {
        v4 = v2[3];
        if ( v4 != v2[2] )
          _libc_free(v4);
      }
      v2 += 8;
    }
    while ( v3 != v2 );
  }
}
