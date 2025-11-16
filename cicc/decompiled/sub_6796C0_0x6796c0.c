// Function: sub_6796C0
// Address: 0x6796c0
//
__int64 sub_6796C0()
{
  __int64 v0; // rdi
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r12

  v0 = 0;
  v1 = 1;
  v4 = sub_65CFF0(0, 1);
  if ( word_4F06418[0] == 9
    || (v1 = (__int64)&dword_4F063F8, v0 = 253, sub_6851C0(253, &dword_4F063F8), word_4F06418[0] == 9) )
  {
    sub_7B8B50(v0, v1, v2, v3);
    return v4;
  }
  else
  {
    do
      sub_7B8B50(253, &dword_4F063F8, v2, v3);
    while ( word_4F06418[0] != 9 );
    sub_7B8B50(253, &dword_4F063F8, v2, v3);
    return v4;
  }
}
