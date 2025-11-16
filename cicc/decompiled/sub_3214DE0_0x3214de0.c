// Function: sub_3214DE0
// Address: 0x3214de0
//
__int64 __fastcall sub_3214DE0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi

  v3 = a1[3];
  v4 = a1[4];
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(*(_QWORD *)v3 + 16LL);
      if ( v5 != *(_QWORD *)v3 + 32LL )
        _libc_free(v5);
      v3 += 8;
    }
    while ( v4 != v3 );
    v4 = a1[3];
  }
  if ( v4 )
  {
    a2 = a1[5] - v4;
    j_j___libc_free_0(v4);
  }
  return sub_C65770(a1 + 1, a2);
}
