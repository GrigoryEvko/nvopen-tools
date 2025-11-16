// Function: sub_679730
// Address: 0x679730
//
__int64 __fastcall sub_679730(__int64 a1, unsigned int a2, unsigned int *a3)
{
  __int64 v3; // rcx
  unsigned int *v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r12

  v3 = a2;
  v4 = a3;
  v7 = sub_7C68A0(a1, a3, 1, v3);
  if ( word_4F06418[0] == 9 || (v4 = &dword_4F063F8, a1 = 253, sub_6851C0(253, &dword_4F063F8), word_4F06418[0] == 9) )
  {
    sub_7B8B50(a1, v4, v5, v6);
    return v7;
  }
  else
  {
    do
      sub_7B8B50(253, &dword_4F063F8, v5, v6);
    while ( word_4F06418[0] != 9 );
    sub_7B8B50(253, &dword_4F063F8, v5, v6);
    return v7;
  }
}
