// Function: sub_70BF90
// Address: 0x70bf90
//
_BOOL8 __fastcall sub_70BF90(
        unsigned __int8 a1,
        const __m128i *a2,
        const __m128i *a3,
        _OWORD *a4,
        _DWORD *a5,
        _DWORD *a6)
{
  _BOOL8 result; // rax
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v12[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_70B8D0(a1, a2, a3, a4, &v11, v12);
  sub_70B8D0(a1, a2 + 1, a3 + 1, a4 + 1, (_DWORD *)&v11 + 1, (_DWORD *)v12 + 1);
  *a5 = v11 != 0;
  result = v12[0] != 0;
  *a6 = result;
  return result;
}
