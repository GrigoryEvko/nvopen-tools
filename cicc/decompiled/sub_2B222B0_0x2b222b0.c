// Function: sub_2B222B0
// Address: 0x2b222b0
//
unsigned __int64 __fastcall sub_2B222B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // of
  unsigned __int64 result; // rax

  v2 = sub_2B21F70(*(_QWORD *)a1, **(_QWORD **)(a1 + 8), 0);
  v3 = __OFADD__(a2, v2);
  result = a2 + v2;
  if ( v3 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( a2 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
