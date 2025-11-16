// Function: sub_18F2920
// Address: 0x18f2920
//
__int64 __fastcall sub_18F2920(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v4 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9B6E8, 1u);
  if ( v4 && (v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_4F9B6E8)) != 0 )
    v6 = v5 + 360;
  else
    v6 = 0;
  return sub_18F25B0(a2, v6);
}
