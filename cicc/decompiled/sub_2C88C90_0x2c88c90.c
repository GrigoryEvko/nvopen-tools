// Function: sub_2C88C90
// Address: 0x2c88c90
//
__int64 __fastcall sub_2C88C90(__int64 a1)
{
  __int64 v1; // rsi

  v1 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = off_49D3FA0;
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8 * v1, 8);
}
