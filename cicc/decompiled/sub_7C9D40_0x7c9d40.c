// Function: sub_7C9D40
// Address: 0x7c9d40
//
const char *__fastcall sub_7C9D40()
{
  __int64 v0; // rbp
  _QWORD v2[7]; // [rsp-38h] [rbp-38h] BYREF

  if ( word_4F06418[0] == 1 )
    return *(const char **)(qword_4D04A00 + 8);
  if ( (unsigned __int16)(word_4F06418[0] - 16) <= 1u || !word_4F06418[0] )
    return "<placeholder error token>";
  v2[6] = v0;
  sub_7ADF70((__int64)v2, 0);
  sub_7AE360((__int64)v2);
  sub_7C9CD0((int *)&dword_4F063F8, 0);
  sub_7C9730((__int64)v2);
  return sub_7C9CF0();
}
