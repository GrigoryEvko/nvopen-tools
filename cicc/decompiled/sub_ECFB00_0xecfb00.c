// Function: sub_ECFB00
// Address: 0xecfb00
//
__int64 __fastcall sub_ECFB00(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi

  *(_QWORD *)(a1 + 24) = a2;
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 40LL))(a2);
  v4 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 32) = v3;
  sub_ECE3C0(a1, v4);
  (*(void (__fastcall **)(_QWORD, const char *, __int64, __int64, __int64 (*)()))(**(_QWORD **)(a1 + 8) + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".text",
    5,
    a1,
    sub_ECE6D0);
  (*(void (__fastcall **)(_QWORD, const char *, __int64, __int64, __int64 (__fastcall *)(__int64)))(**(_QWORD **)(a1 + 8)
                                                                                                  + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".data",
    5,
    a1,
    sub_ECE720);
  (*(void (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64 *, __int64, __int64, __int64)))(**(_QWORD **)(a1 + 8) + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".section",
    8,
    a1,
    sub_ECFAA0);
  (*(void (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64, __int64, __int64, __int64)))(**(_QWORD **)(a1 + 8) + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".size",
    5,
    a1,
    sub_ECED10);
  (*(void (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64)))(**(_QWORD **)(a1 + 8)
                                                                                            + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".type",
    5,
    a1,
    sub_ECEEB0);
  (*(void (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64)))(**(_QWORD **)(a1 + 8)
                                                                                            + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".ident",
    6,
    a1,
    sub_ECE770);
  (*(void (__fastcall **)(_QWORD, const char *, __int64, __int64, __int64 (__fastcall *)(__int64, __int64, __int64)))(**(_QWORD **)(a1 + 8) + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".weak",
    5,
    a1,
    sub_ECE880);
  (*(void (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64, __int64, __int64)))(**(_QWORD **)(a1 + 8) + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".local",
    6,
    a1,
    sub_ECE880);
  (*(void (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64, __int64, __int64)))(**(_QWORD **)(a1 + 8) + 16LL))(
    *(_QWORD *)(a1 + 8),
    ".internal",
    9,
    a1,
    sub_ECE880);
  return (*(__int64 (__fastcall **)(_QWORD, char *, __int64, __int64, __int64 (__fastcall *)(__int64, __int64, __int64)))(**(_QWORD **)(a1 + 8) + 16LL))(
           *(_QWORD *)(a1 + 8),
           ".hidden",
           7,
           a1,
           sub_ECE880);
}
