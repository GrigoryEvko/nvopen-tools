// Function: sub_1553920
// Address: 0x1553920
//
char __fastcall sub_1553920(__int64 *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  char result; // al

  if ( a3 )
    return sub_15535E0(a1, a2, a3, a4);
  v6 = *(_QWORD *)(a4 + 16);
  v7 = sub_154BC70(a4);
  result = sub_1553590((__int64)a1, a2, v7, v6);
  if ( !result )
    return sub_15535E0(a1, a2, a3, a4);
  return result;
}
