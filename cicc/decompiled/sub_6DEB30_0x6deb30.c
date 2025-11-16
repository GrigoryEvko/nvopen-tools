// Function: sub_6DEB30
// Address: 0x6deb30
//
__int64 sub_6DEB30()
{
  __int64 result; // rax
  __int64 v1; // rcx

  result = 0;
  if ( (*(_BYTE *)(qword_4D03C50 + 20LL) & 1) == 0 )
  {
    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0
      || ((v1 = qword_4F04C68[0] + 776LL * dword_4F04C64, dword_4F04C44 != -1)
       || (*(_BYTE *)(v1 + 6) & 2) != 0
       || (result = 1, *(_BYTE *)(qword_4D03C50 + 16LL) <= 1u))
      && (result = 1, *(_BYTE *)(v1 + 4) != 8) )
    {
      result = 0;
      if ( *(char *)(qword_4D03C50 + 19LL) < 0 )
      {
        result = 1;
        if ( !unk_4D048AC )
        {
          result = dword_4F077BC;
          if ( dword_4F077BC )
          {
            result = 0;
            if ( !(_DWORD)qword_4F077B4 )
              return qword_4F077A8 <= 0x15F8Fu;
          }
        }
      }
    }
  }
  return result;
}
