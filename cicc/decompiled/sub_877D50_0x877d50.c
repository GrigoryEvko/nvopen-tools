// Function: sub_877D50
// Address: 0x877d50
//
__int64 __fastcall sub_877D50(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 89) & 8) == 0 && (*(_BYTE *)(a2 + 73) & 1) == 0 )
  {
    result = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 + 8) = result;
  }
  return result;
}
