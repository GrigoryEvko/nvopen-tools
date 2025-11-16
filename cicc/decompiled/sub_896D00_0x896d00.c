// Function: sub_896D00
// Address: 0x896d00
//
__int64 __fastcall sub_896D00(__int64 a1, char a2)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    if ( dword_4F04C44 != -1 || (result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 6) & 2) != 0) )
    {
      result = (unsigned int)*(unsigned __int8 *)(a1 + 80) - 4;
      if ( (unsigned __int8)(*(_BYTE *)(a1 + 80) - 4) <= 1u )
      {
        result = *(_QWORD *)(a1 + 88);
        if ( (*(_BYTE *)(result + 177) & 0x10) != 0 )
        {
          result = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
          *(_BYTE *)(result + 264) = a2;
        }
      }
    }
  }
  return result;
}
