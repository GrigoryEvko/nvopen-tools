// Function: sub_68B6B0
// Address: 0x68b6b0
//
__int64 __fastcall sub_68B6B0(__int64 a1)
{
  __int64 result; // rax

  result = dword_4F04C58;
  if ( dword_4F04C58 != -1 )
  {
    result = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
    if ( *(char *)(result + 192) < 0
      && (*(_BYTE *)(result + 202) & 2) == 0
      && !unk_4D04370
      && *(_BYTE *)(result + 172) != 2 )
    {
      result = *(_QWORD *)(a1 + 88);
      if ( *(_BYTE *)(a1 + 80) == 7 )
      {
        if ( *(_BYTE *)(result + 136) == 2 && (*(_BYTE *)(result + 89) & 1) == 0 )
          return sub_85E9E0(0, dword_4F07508);
      }
      else if ( *(_BYTE *)(result + 172) == 2 )
      {
        return sub_85E9E0(0, dword_4F07508);
      }
    }
  }
  return result;
}
