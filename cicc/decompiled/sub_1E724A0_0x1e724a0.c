// Function: sub_1E724A0
// Address: 0x1e724a0
//
__int64 __fastcall sub_1E724A0(unsigned __int64 *a1)
{
  _QWORD *v2; // rdi
  unsigned __int64 v3; // rdi

  v2 = a1 + 64;
  *(v2 - 64) = &unk_49FC9E0;
  sub_1E72310(v2);
  sub_1E72310(a1 + 18);
  v3 = a1[6];
  if ( (unsigned __int64 *)v3 != a1 + 8 )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 976);
}
