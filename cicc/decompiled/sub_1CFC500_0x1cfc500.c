// Function: sub_1CFC500
// Address: 0x1cfc500
//
__int64 __fastcall sub_1CFC500(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi

  *a1 = off_49F93A0;
  v2 = a1[105];
  if ( v2 )
    j_j___libc_free_0(v2, a1[107] - v2);
  v3 = a1[102];
  if ( v3 )
    j_j___libc_free_0(v3, a1[104] - v3);
  v4 = a1[83];
  if ( (_QWORD *)v4 != a1 + 85 )
    _libc_free(v4);
  v5 = a1[80];
  *a1 = &unk_49F9818;
  if ( v5 )
    j_j___libc_free_0(v5, a1[82] - v5);
  return sub_1F012F0(a1);
}
