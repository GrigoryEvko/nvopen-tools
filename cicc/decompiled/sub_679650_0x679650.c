// Function: sub_679650
// Address: 0x679650
//
__int64 __fastcall sub_679650(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx

  sub_6D6050(a1, a2, 0, 0);
  if ( word_4F06418[0] != 9 )
  {
    a2 = &dword_4F063F8;
    a1 = 253;
    sub_6851C0(253, &dword_4F063F8);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(253, &dword_4F063F8, v2, v3);
  }
  return sub_7B8B50(a1, a2, v2, v3);
}
