// Function: sub_6EFF80
// Address: 0x6eff80
//
__int64 sub_6EFF80()
{
  __int64 v0; // rdi

  if ( unk_4D0439C )
    return sub_72C390();
  if ( qword_4D03C50 && !*(_BYTE *)(qword_4D03C50 + 16LL) )
  {
    v0 = 7;
    if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
      return sub_72BA30(unk_4F06AC9);
  }
  else
  {
    v0 = 5;
  }
  return sub_72BA30(v0);
}
