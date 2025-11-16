// Function: sub_2253FC0
// Address: 0x2253fc0
//
void __fastcall sub_2253FC0(_QWORD *a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbp

  v2 = (unsigned __int64 *)a1[6];
  v3 = a1[7];
  if ( v2 != (unsigned __int64 *)v3 )
  {
    do
    {
      v4 = *v2;
      if ( *v2 )
      {
        _libc_free(*(_QWORD *)(v4 + 8));
        sub_2209150((volatile signed __int32 **)(v4 + 16));
        j___libc_free_0(v4);
      }
      ++v2;
    }
    while ( (unsigned __int64 *)a1[7] != v2 );
    v3 = a1[6];
  }
  if ( v3 )
    j___libc_free_0(v3);
}
