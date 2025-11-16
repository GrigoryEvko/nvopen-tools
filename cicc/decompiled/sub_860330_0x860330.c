// Function: sub_860330
// Address: 0x860330
//
__int64 __fastcall sub_860330(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 )
  {
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 40);
    result = (*(_BYTE *)(a2 + 88) >> 4) & 7;
    *(_BYTE *)(a1 + 40) = result;
  }
  return result;
}
