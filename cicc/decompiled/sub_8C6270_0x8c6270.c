// Function: sub_8C6270
// Address: 0x8c6270
//
__int64 __fastcall sub_8C6270(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; result; result = *(_QWORD *)(result + 112) )
  {
    if ( ((*(_BYTE *)(result + 193) & 0x10) == 0 || *(_BYTE *)(result + 174) != 7 && !*(_QWORD *)(result + 280))
      && ((*(_BYTE *)(result + 195) & 0x21) != 1 || !*(_QWORD *)(result + 240)) )
    {
      break;
    }
  }
  return result;
}
