// Function: sub_87D420
// Address: 0x87d420
//
__int64 __fastcall sub_87D420(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 i; // rdi

  qword_4F5FF90 = a3;
  qword_4F5FF88 = 0;
  sub_7461E0((__int64)qword_4F5FEE0);
  qword_4F5FEE0[0] = (__int64)sub_876ED0;
  byte_4F5FF69 = dword_4F077C4 == 1;
  if ( dword_4D0460C )
    byte_4F5FF78 = a4;
  qword_4F5FF98 = a1;
  sub_87D380(a2, (void (__fastcall **)(const char *, _QWORD))qword_4F5FEE0);
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 80) - 10) > 1u )
    return a1;
  for ( i = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  sub_74BA50(i, (__int64)qword_4F5FEE0);
  return a1;
}
