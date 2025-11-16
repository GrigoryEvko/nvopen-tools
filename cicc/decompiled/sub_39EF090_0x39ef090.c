// Function: sub_39EF090
// Address: 0x39ef090
//
__int64 __fastcall sub_39EF090(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  sub_390D5F0(a1[33], a2, 0);
  sub_38E2920(a2, 0);
  *(_BYTE *)(a2 + 8) &= ~0x10u;
  return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD))(*a1 + 368))(a1, a2, a3, a4);
}
