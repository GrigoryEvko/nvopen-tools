// Function: sub_39EF550
// Address: 0x39ef550
//
void __fastcall sub_39EF550(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A40B50;
  v2 = a1[41];
  if ( (_QWORD *)v2 != a1 + 43 )
    _libc_free(v2);
  sub_38D40E0(a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
