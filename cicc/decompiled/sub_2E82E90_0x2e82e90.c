// Function: sub_2E82E90
// Address: 0x2e82e90
//
void __fastcall sub_2E82E90(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(unsigned __int64 **)(a1 + 80);
  *(_QWORD *)a1 = &unk_49D9D40;
  v3 = &v2[10 * *(unsigned int *)(a1 + 88)];
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 10;
      v4 = v3[4];
      if ( (unsigned __int64 *)v4 != v3 + 6 )
        j_j___libc_free_0(v4);
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3);
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 80);
  }
  if ( v3 != (unsigned __int64 *)(a1 + 96) )
    _libc_free((unsigned __int64)v3);
}
