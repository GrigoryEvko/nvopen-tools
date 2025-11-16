// Function: sub_2BF05A0
// Address: 0x2bf05a0
//
__int64 __fastcall sub_2BF05A0(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rdi

  result = *(_QWORD *)(a1 + 120);
  for ( i = a1 + 112; result != i; result = *(_QWORD *)(result + 8) )
  {
    if ( !result )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(result - 16) - 27 > 9 )
      break;
  }
  return result;
}
