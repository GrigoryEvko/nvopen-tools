// Function: sub_86F930
// Address: 0x86f930
//
__int64 sub_86F930()
{
  _DWORD *v0; // rax

  v0 = (_DWORD *)(qword_4D03B98 + 176LL * unk_4D03B90);
  if ( (_DWORD *)qword_4D03B98 == v0 )
    return 0;
  while ( *v0 != 3 )
  {
    v0 -= 44;
    if ( (_DWORD *)qword_4D03B98 == v0 )
      return 0;
  }
  return 1;
}
