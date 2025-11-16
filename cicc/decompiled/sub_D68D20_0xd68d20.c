// Function: sub_D68D20
// Address: 0xd68d20
//
unsigned __int8 __fastcall sub_D68D20(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int8 result; // al

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)a1 = 2LL * a2;
  result = a3 != -4096;
  if ( (result & (a3 != 0)) != 0 && a3 != -8192 )
    return sub_BD73F0(a1);
  return result;
}
