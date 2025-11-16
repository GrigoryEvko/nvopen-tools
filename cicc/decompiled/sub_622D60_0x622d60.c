// Function: sub_622D60
// Address: 0x622d60
//
__int64 __fastcall sub_622D60(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(result + 80) - 10) <= 1u )
      *(_BYTE *)(result + 104) |= 2u;
  }
  return result;
}
