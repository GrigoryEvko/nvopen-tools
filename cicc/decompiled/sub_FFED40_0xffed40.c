// Function: sub_FFED40
// Address: 0xffed40
//
_BOOL8 __fastcall sub_FFED40(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 v8; // r15
  int v9; // r13d

  if ( (unsigned __int8)(*(_BYTE *)a1 - 82) > 1u )
    return 0;
  v7 = *(_QWORD *)(a1 - 64);
  v8 = *(_QWORD *)(a1 - 32);
  v9 = *(_WORD *)(a1 + 2) & 0x3F;
  return v9 == a2 && v7 == a3 && v8 == a4 || v9 == (unsigned int)sub_B52F50(a2) && v7 == a4 && v8 == a3;
}
