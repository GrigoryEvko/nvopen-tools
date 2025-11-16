// Function: sub_ECFD30
// Address: 0xecfd30
//
__int64 __fastcall sub_ECFD30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi

  *(_QWORD *)(a1 + 24) = a2;
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL))(a2);
  v3 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 32) = v2;
  sub_ECE3C0(a1, v3);
  return (*(__int64 (__fastcall **)(_QWORD, const char *, __int64, __int64, void (__noreturn *)()))(**(_QWORD **)(a1 + 8)
                                                                                                  + 16LL))(
           *(_QWORD *)(a1 + 8),
           ".csect",
           6,
           a1,
           sub_ECFC80);
}
