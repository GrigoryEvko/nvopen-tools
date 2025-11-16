// Function: sub_301F4E0
// Address: 0x301f4e0
//
void __fastcall sub_301F4E0(__int64 a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r12

  v1 = *(unsigned __int64 **)(a1 + 16);
  v2 = &v1[4 * *(unsigned int *)(a1 + 24)];
  *(_QWORD *)a1 = &unk_4A2E2B0;
  if ( v1 != v2 )
  {
    do
    {
      v2 -= 4;
      if ( (unsigned __int64 *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2);
    }
    while ( v1 != v2 );
    v2 = *(unsigned __int64 **)(a1 + 16);
  }
  if ( v2 != (unsigned __int64 *)(a1 + 32) )
    _libc_free((unsigned __int64)v2);
  nullsub_339();
}
