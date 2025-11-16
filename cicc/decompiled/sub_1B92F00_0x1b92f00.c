// Function: sub_1B92F00
// Address: 0x1b92f00
//
bool __fastcall sub_1B92F00(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  bool result; // al

  v2 = sub_13A4950(a2);
  result = 0;
  if ( v2 )
    return (unsigned int)sub_1BF20B0(*(_QWORD *)(a1 + 320), v2) != 0;
  return result;
}
