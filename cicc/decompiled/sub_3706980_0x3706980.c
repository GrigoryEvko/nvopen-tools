// Function: sub_3706980
// Address: 0x3706980
//
void __fastcall sub_3706980(_QWORD *a1)
{
  _QWORD *v1; // r12
  unsigned __int64 v2; // rdi
  volatile signed __int32 *v3; // rdi

  v1 = (_QWORD *)a1[1];
  *a1 = &unk_4A3C658;
  if ( v1 )
  {
    v2 = v1[14];
    if ( (_QWORD *)v2 != v1 + 16 )
      _libc_free(v2);
    v3 = (volatile signed __int32 *)v1[6];
    v1[4] = &unk_49E6870;
    if ( v3 )
      sub_A191D0(v3);
    j_j___libc_free_0((unsigned __int64)v1);
  }
}
