// Function: sub_D4B000
// Address: 0xd4b000
//
__int64 __fastcall sub_D4B000(__int64 *a1)
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
