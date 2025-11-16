// Function: sub_6E5200
// Address: 0x6e5200
//
__int64 __fastcall sub_6E5200(unsigned __int8 a1, int a2, int a3, int a4, int a5, __int64 a6, __int64 a7, __int64 a8)
{
  unsigned int *v8; // r11
  __int64 result; // rax
  unsigned int v10; // [rsp+Ch] [rbp-4h] BYREF

  v10 = 0;
  v8 = &v10;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v8 = 0;
  sub_713ED0(
    a1,
    a2,
    a3,
    a4,
    a5,
    *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u,
    *(_BYTE *)(qword_4D03C50 + 17LL) & 1,
    a6,
    a7,
    (__int64)v8,
    a8);
  result = v10;
  if ( v10 )
    return sub_6E50A0();
  return result;
}
