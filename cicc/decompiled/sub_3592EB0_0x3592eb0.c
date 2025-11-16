// Function: sub_3592EB0
// Address: 0x3592eb0
//
void __fastcall sub_3592EB0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi

  *a1 = &unk_4A39A50;
  v2 = a1[10];
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = (unsigned __int64 *)a1[7];
  v4 = (unsigned __int64 *)a1[6];
  *a1 = &unk_4A399F0;
  if ( v3 != v4 )
  {
    do
    {
      if ( *v4 )
        j_j___libc_free_0(*v4);
      v4 += 3;
    }
    while ( v3 != v4 );
    v4 = (unsigned __int64 *)a1[6];
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  v5 = a1[3];
  if ( v5 )
    j_j___libc_free_0(v5);
  j_j___libc_free_0((unsigned __int64)a1);
}
