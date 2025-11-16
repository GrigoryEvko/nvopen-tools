// Function: sub_2F510F0
// Address: 0x2f510f0
//
__int64 __fastcall sub_2F510F0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 result; // rax

  result = 0;
  if ( *(unsigned __int8 *)(*(_QWORD *)(a1 + 72) + a3) < a2 )
  {
    result = 1;
    if ( a2 == 1 )
      return (unsigned int)sub_2F50F60(a1, a3) ^ 1;
  }
  return result;
}
