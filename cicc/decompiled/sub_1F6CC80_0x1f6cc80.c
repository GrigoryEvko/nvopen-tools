// Function: sub_1F6CC80
// Address: 0x1f6cc80
//
char __fastcall sub_1F6CC80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 v4; // rsi

  v3 = *(_QWORD *)(*(_QWORD *)a3 + 88LL);
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
  if ( *(_DWORD *)(v3 + 32) <= 0x40u )
    return (*(_QWORD *)(v3 + 24) & ~*(_QWORD *)(v4 + 24)) == 0;
  else
    return sub_16A5A00((__int64 *)(v3 + 24), (__int64 *)(v4 + 24));
}
