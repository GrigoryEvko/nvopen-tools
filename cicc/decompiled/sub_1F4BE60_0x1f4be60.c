// Function: sub_1F4BE60
// Address: 0x1f4be60
//
__int64 __fastcall sub_1F4BE60(__int64 a1)
{
  __int64 result; // rax

  result = sub_38D72B0(*(_QWORD *)(a1 + 176));
  if ( (int)result < 0 )
    return 1000;
  return result;
}
