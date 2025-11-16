// Function: sub_1452BC0
// Address: 0x1452bc0
//
__int64 __fastcall sub_1452BC0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rsi

  result = a1;
  if ( *(_WORD *)(a1 + 24) == 4 && *(_QWORD *)(a1 + 40) == 2 )
  {
    v2 = *(_QWORD **)(a1 + 32);
    if ( !*(_WORD *)(*v2 + 24LL) )
      return v2[1];
  }
  return result;
}
