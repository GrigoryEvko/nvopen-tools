// Function: sub_12E9DC0
// Address: 0x12e9dc0
//
__int64 __fastcall sub_12E9DC0(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64, _QWORD); // rbx
  __int64 v4; // rsi

  v2 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v4 = sub_14A7550();
  return v2(a2, v4, 0);
}
