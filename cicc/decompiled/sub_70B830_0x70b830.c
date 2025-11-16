// Function: sub_70B830
// Address: 0x70b830
//
__int64 __fastcall sub_70B830(unsigned __int8 a1, const __m128i *a2, __int64 *a3, _DWORD *a4, _DWORD *a5)
{
  __int64 v7; // rdx
  __int64 result; // rax
  _QWORD v9[6]; // [rsp+0h] [rbp-30h] BYREF

  *a4 = 0;
  *a5 = 0;
  v9[0] = sub_709B30(a1, a2);
  v9[1] = v7;
  unk_4F968EA = 0;
  result = sub_12F9910(v9, 0);
  *a3 = result;
  if ( (unk_4F968EA & 0x10) != 0 )
    *a4 = 1;
  return result;
}
