// Function: sub_B18F80
// Address: 0xb18f80
//
__int64 __fastcall sub_B18F80(__int64 a1)
{
  __int64 result; // rax

  for ( result = *(_QWORD *)(a1 + 16); result; result = *(_QWORD *)(result + 8) )
  {
    if ( (unsigned __int8)(**(_BYTE **)(result + 24) - 30) <= 0xAu )
      break;
  }
  return result;
}
