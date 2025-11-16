// Function: sub_31121F0
// Address: 0x31121f0
//
void __fastcall sub_31121F0(unsigned __int64 **a1)
{
  unsigned __int64 *v1; // r12
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdi

  v1 = *a1;
  if ( *a1 )
  {
    v2 = v1[1];
    if ( v2 )
      sub_3111400(v2);
    v3 = *v1;
    if ( *v1 )
    {
      sub_3112140(v3 + 16);
      v4 = *(_QWORD *)(v3 + 16);
      if ( v4 != v3 + 64 )
        j_j___libc_free_0(v4);
      j_j___libc_free_0(v3);
    }
    j_j___libc_free_0((unsigned __int64)v1);
  }
}
