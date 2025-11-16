// Function: sub_5E4AC0
// Address: 0x5e4ac0
//
__int64 __fastcall sub_5E4AC0(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( !*(_QWORD *)a1 )
  {
    if ( (*(_BYTE *)(a1 + 24) & 0x10) == 0 )
      return *(_QWORD *)(a1 + 16) != 0;
    result = dword_4F077BC;
    if ( dword_4F077BC )
    {
      result = dword_4F077B4;
      if ( dword_4F077B4 )
        return 0;
      if ( qword_4F077A8 <= 0xEB27u )
        return *(_QWORD *)(a1 + 16) != 0;
    }
  }
  return result;
}
