// Function: sub_2F06340
// Address: 0x2f06340
//
void __fastcall sub_2F06340(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A2A480;
  v2 = a1[1];
  if ( v2 )
    j_j___libc_free_0(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
