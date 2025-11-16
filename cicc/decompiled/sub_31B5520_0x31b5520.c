// Function: sub_31B5520
// Address: 0x31b5520
//
__int64 __fastcall sub_31B5520(
        __int64 a1,
        void (__fastcall **a2)(__int64, _QWORD, _QWORD, _QWORD, _QWORD),
        _QWORD *a3,
        _QWORD *a4)
{
  (*a2)(a1, *a3, a3[1], *a4, a4[1]);
  return a1;
}
