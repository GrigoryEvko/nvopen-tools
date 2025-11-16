// Function: sub_7BC0A0
// Address: 0x7bc0a0
//
__int64 __fastcall sub_7BC0A0(_DWORD *a1)
{
  __int64 result; // rax

  if ( word_4F06418[0] == 42 )
  {
    if ( !dword_4F07770 )
    {
      result = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( *(_QWORD *)(result + 632) <= 1u )
        goto LABEL_2;
      if ( *a1 )
        return result;
      sub_6851C0(0x364u, dword_4F07508);
      *a1 = 1;
    }
    return sub_7BC010();
  }
LABEL_2:
  if ( !*a1 )
  {
    sub_6851D0(0x1B7u);
    result = dword_4F07770;
    if ( dword_4F07770 )
    {
      if ( word_4F06418[0] == 42 )
        result = sub_7BC010();
    }
    *a1 = 1;
  }
  return result;
}
