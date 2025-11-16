// Function: sub_2D63A10
// Address: 0x2d63a10
//
__int64 __fastcall sub_2D63A10(__int64 *a1)
{
  __int64 result; // rax

  for ( result = *a1; result; *a1 = result )
  {
    if ( (unsigned __int8)(**(_BYTE **)(result + 24) - 30) <= 0xAu )
      break;
    result = *(_QWORD *)(result + 8);
  }
  return result;
}
