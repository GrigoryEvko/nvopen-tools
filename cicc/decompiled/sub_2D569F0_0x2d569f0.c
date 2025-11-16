// Function: sub_2D569F0
// Address: 0x2d569f0
//
void __fastcall sub_2D569F0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1 + 17;
  v3 = a1[15];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[12];
  if ( (_QWORD *)v4 != a1 + 14 )
    _libc_free(v4);
  v5 = a1[2];
  if ( (_QWORD *)v5 != a1 + 4 )
    _libc_free(v5);
}
