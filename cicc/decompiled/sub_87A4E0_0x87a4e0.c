// Function: sub_87A4E0
// Address: 0x87a4e0
//
__int64 __fastcall sub_87A4E0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 32);
  if ( !unk_4D04968 )
    result = *(_QWORD *)(a1 + 24);
  for ( ; result; result = *(_QWORD *)(result + 8) )
  {
    if ( *(_BYTE *)(result + 80) == 1 )
      break;
  }
  return result;
}
