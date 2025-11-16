// Function: sub_325F870
// Address: 0x325f870
//
char __fastcall sub_325F870(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 v4; // rsi

  v3 = *(_QWORD *)(*(_QWORD *)a3 + 96LL);
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  if ( *(_DWORD *)(v3 + 32) <= 0x40u )
    return (*(_QWORD *)(v3 + 24) & ~*(_QWORD *)(v4 + 24)) == 0;
  else
    return sub_C446F0((__int64 *)(v3 + 24), (__int64 *)(v4 + 24));
}
