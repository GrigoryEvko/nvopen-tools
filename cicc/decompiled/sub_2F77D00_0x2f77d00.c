// Function: sub_2F77D00
// Address: 0x2f77d00
//
__int64 __fastcall sub_2F77D00(__int64 a1, int a2, __int64 a3)
{
  return sub_2F73FB0(
           *(_QWORD *)(a1 + 32),
           *(_QWORD *)(a1 + 24),
           *(_BYTE *)(a1 + 58),
           a2,
           a3 & 0xFFFFFFFFFFFFFFF8LL,
           (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2F745D0,
           0,
           0);
}
