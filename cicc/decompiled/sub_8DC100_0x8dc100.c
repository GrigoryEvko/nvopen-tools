// Function: sub_8DC100
// Address: 0x8dc100
//
__int64 __fastcall sub_8DC100(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( (*(_BYTE *)(a1 + 142) & 2) != 0 )
    {
      return *(_BYTE *)(a1 + 142) & 1;
    }
    else
    {
      qword_4F60580 = 0;
      qword_4F60578 = 0;
      dword_4F60570 = 0;
      dword_4F6056C = 1;
      dword_4F60568 = 0;
      result = sub_8D9600(a1, (__int64 (__fastcall *)(__int64, unsigned int *))sub_8DB8B0, 0x1E07u);
      *(_BYTE *)(a1 + 142) = *(_BYTE *)(a1 + 142) & 0xFC | result & 1 | 2;
    }
  }
  return result;
}
