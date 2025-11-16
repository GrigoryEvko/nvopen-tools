// Function: sub_305DB70
// Address: 0x305db70
//
void __fastcall sub_305DB70(__int64 a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r12

  v1 = *(unsigned __int64 **)(a1 + 8);
  v2 = &v1[4 * *(unsigned int *)(a1 + 16)];
  *(_QWORD *)a1 = &unk_4A30878;
  if ( v1 != v2 )
  {
    do
    {
      v2 -= 4;
      if ( (unsigned __int64 *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2);
    }
    while ( v1 != v2 );
    v2 = *(unsigned __int64 **)(a1 + 8);
  }
  if ( v2 != (unsigned __int64 *)(a1 + 24) )
    _libc_free((unsigned __int64)v2);
  nullsub_1602();
}
