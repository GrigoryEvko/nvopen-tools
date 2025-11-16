// Function: sub_2306600
// Address: 0x2306600
//
void __fastcall sub_2306600(_QWORD *a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r12

  v1 = (unsigned __int64 *)a1[2];
  v2 = (unsigned __int64 *)a1[1];
  *a1 = &unk_4A0D278;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        j_j___libc_free_0(*v2);
      v2 += 3;
    }
    while ( v1 != v2 );
    v2 = (unsigned __int64 *)a1[1];
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
}
