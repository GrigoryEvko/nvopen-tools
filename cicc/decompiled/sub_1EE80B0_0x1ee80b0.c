// Function: sub_1EE80B0
// Address: 0x1ee80b0
//
__int64 __fastcall sub_1EE80B0(__int64 a1, int a2, __int64 a3)
{
  return sub_1EE4FF0(
           *(_QWORD *)(a1 + 32),
           *(_QWORD *)(a1 + 24),
           *(_BYTE *)(a1 + 58),
           a2,
           a3 & 0xFFFFFFFFFFFFFFF8LL,
           0,
           (unsigned __int8 (__fastcall *)(__int64, __int64))sub_1EE5700);
}
