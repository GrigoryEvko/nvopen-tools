// Function: sub_2E311E0
// Address: 0x2e311e0
//
__int64 __fastcall sub_2E311E0(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rdi

  result = *(_QWORD *)(a1 + 56);
  for ( i = a1 + 48; i != result; result = *(_QWORD *)(result + 8) )
  {
    if ( *(_WORD *)(result + 68) != 68 && *(_WORD *)(result + 68) )
      break;
  }
  return result;
}
