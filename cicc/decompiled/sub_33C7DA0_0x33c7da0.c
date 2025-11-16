// Function: sub_33C7DA0
// Address: 0x33c7da0
//
__int64 __fastcall sub_33C7DA0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 result; // rax

  v2 = *(__int64 **)(a1 + 24);
  for ( result = *v2; **(_QWORD **)(a1 + 32) != *v2; result = *v2 )
  {
    if ( a2 != *(_QWORD *)(result + 16) )
      break;
    *v2 = *(_QWORD *)(result + 32);
    v2 = *(__int64 **)(a1 + 24);
  }
  return result;
}
