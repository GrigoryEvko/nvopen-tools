// Function: sub_13CB8F0
// Address: 0x13cb8f0
//
_BOOL8 __fastcall sub_13CB8F0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 v8; // r15
  int v9; // r12d

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 16) - 75) > 1u )
    return 0;
  v7 = *(_QWORD *)(a1 - 48);
  v8 = *(_QWORD *)(a1 - 24);
  v9 = *(_WORD *)(a1 + 18) & 0x7FFF;
  return a4 == v8 && a3 == v7 && a2 == v9 || v9 == (unsigned int)sub_15FF5D0(a2) && a4 == v7 && a3 == v8;
}
