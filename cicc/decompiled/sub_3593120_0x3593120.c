// Function: sub_3593120
// Address: 0x3593120
//
void __fastcall sub_3593120(_QWORD *a1)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi

  *a1 = off_4A39A20;
  v2 = a1[6];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = (unsigned __int64 *)a1[4];
  v4 = (unsigned __int64 *)a1[3];
  if ( v3 != v4 )
  {
    do
    {
      v5 = v4[5];
      if ( v5 )
        j_j___libc_free_0(v5);
      if ( (unsigned __int64 *)*v4 != v4 + 2 )
        j_j___libc_free_0(*v4);
      v4 += 10;
    }
    while ( v3 != v4 );
    v4 = (unsigned __int64 *)a1[3];
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  j_j___libc_free_0((unsigned __int64)a1);
}
