// Function: sub_39614F0
// Address: 0x39614f0
//
void __fastcall sub_39614F0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v2 = a1 + 28;
  v3 = a1[26];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[12];
  if ( v4 != a1[11] )
    _libc_free(v4);
  j_j___libc_free_0((unsigned __int64)a1);
}
