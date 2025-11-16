// Function: sub_70C0B0
// Address: 0x70c0b0
//
_BOOL8 __fastcall sub_70C0B0(unsigned __int8 a1, const __m128i *a2, _OWORD *a3, _DWORD *a4, _DWORD *a5)
{
  _BOOL8 result; // rax
  __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v10[7]; // [rsp+8h] [rbp-38h] BYREF

  sub_70BAF0(a1, a2, a3, &v9, v10);
  sub_70BAF0(a1, a2 + 1, a3 + 1, (_DWORD *)&v9 + 1, (_DWORD *)v10 + 1);
  *a4 = v9 != 0;
  result = v10[0] != 0;
  *a5 = result;
  return result;
}
