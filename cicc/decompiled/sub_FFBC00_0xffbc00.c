// Function: sub_FFBC00
// Address: 0xffbc00
//
__int64 __fastcall sub_FFBC00(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( !*(_QWORD *)(a1 + 544) || (result = *(unsigned int *)(a1 + 8), result == *(_QWORD *)(a1 + 528)) )
  {
    if ( !*(_QWORD *)(a1 + 552) )
      return sub_FFBA00(a1, a2);
    result = *(unsigned int *)(a1 + 8);
    if ( result == *(_QWORD *)(a1 + 536) )
      return sub_FFBA00(a1, a2);
  }
  return result;
}
