// Function: sub_1C4A170
// Address: 0x1c4a170
//
__int64 __fastcall sub_1C4A170(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi

  v2 = a1 + 8;
  v3 = a1[6];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[2];
  *a1 = off_4985448;
  return j___libc_free_0(v4);
}
