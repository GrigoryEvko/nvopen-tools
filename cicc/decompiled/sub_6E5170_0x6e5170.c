// Function: sub_6E5170
// Address: 0x6e5170
//
__int64 __fastcall sub_6E5170(int a1, int a2, int a3, int a4, int a5, int a6, int a7, __int64 a8, __int64 a9)
{
  unsigned int *v9; // rbx
  __int64 result; // rax
  unsigned int v11; // [rsp+Ch] [rbp-14h] BYREF

  v9 = &v11;
  v11 = 0;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v9 = 0;
  sub_7115B0(
    a1,
    a2,
    a3,
    *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u,
    *(_BYTE *)(qword_4D03C50 + 17LL) & 1,
    (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0,
    0,
    a4,
    a5,
    a6,
    a7,
    a8,
    (__int64)v9,
    a9);
  result = v11;
  if ( v11 )
    return sub_6E50A0();
  return result;
}
