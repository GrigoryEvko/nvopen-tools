// Function: sub_385A210
// Address: 0x385a210
//
__int64 __fastcall sub_385A210(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v6; // rsi

  v4 = (a1 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  if ( (a1 & 4) != 0 )
    v4 = (a1 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v6 = *(_QWORD *)v4;
  if ( *(_BYTE *)(*(_QWORD *)v4 + 16LL) )
    v6 = 0;
  return sub_38599E0(a1, v6, a2, a3);
}
