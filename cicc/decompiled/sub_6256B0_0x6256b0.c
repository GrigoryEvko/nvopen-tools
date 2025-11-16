// Function: sub_6256B0
// Address: 0x6256b0
//
__int64 __fastcall sub_6256B0(char a1)
{
  __int64 result; // rax
  char v2; // di

  result = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 32);
  if ( result )
  {
    v2 = (a1 & 1) << 6;
    do
    {
      if ( *(_BYTE *)(result + 80) == 18 )
        *(_BYTE *)(result + 83) = v2 | *(_BYTE *)(result + 83) & 0xBF;
      result = *(_QWORD *)(result + 16);
    }
    while ( result );
  }
  return result;
}
