// Function: sub_3706A00
// Address: 0x3706a00
//
void __fastcall sub_3706A00(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v3; // rdi
  volatile signed __int32 *v4; // rdi

  v1 = (_QWORD *)a1[1];
  *a1 = &unk_4A3C658;
  if ( v1 )
  {
    v3 = v1[14];
    if ( (_QWORD *)v3 != v1 + 16 )
      _libc_free(v3);
    v4 = (volatile signed __int32 *)v1[6];
    v1[4] = &unk_49E6870;
    if ( v4 )
      sub_A191D0(v4);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
