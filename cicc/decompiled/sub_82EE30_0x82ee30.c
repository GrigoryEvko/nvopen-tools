// Function: sub_82EE30
// Address: 0x82ee30
//
__int64 sub_82EE30()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = dword_4F077BC;
  if ( dword_4F077BC )
  {
    result = dword_4F07734;
    if ( dword_4F07734 )
    {
      result = 0;
      if ( qword_4D03C50 )
      {
        if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x20) != 0 )
        {
          v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( *(_BYTE *)(v1 + 4) == 8 && (dword_4F04C44 != -1 || (*(_BYTE *)(v1 + 6) & 2) != 0) )
          {
            result = 0;
            if ( unk_4F04C48 != -1 )
              return (*(_BYTE *)(v1 + 6) & 6) == 0;
          }
        }
      }
    }
  }
  return result;
}
