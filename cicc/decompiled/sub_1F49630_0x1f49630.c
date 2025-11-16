// Function: sub_1F49630
// Address: 0x1f49630
//
__int64 __fastcall sub_1F49630(__int64 *a1)
{
  int *v1; // rax
  int v2; // eax
  _DWORD *v3; // rax
  _QWORD *v6; // rax
  __m128i *v7; // rsi
  _QWORD *v8; // rsi
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v10)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]

  v1 = (int *)sub_16D40F0((__int64)qword_4FBB3B0);
  if ( v1 )
    v2 = *v1;
  else
    v2 = qword_4FBB3B0[2];
  if ( v2 != 4 )
  {
    v3 = sub_16D40F0((__int64)qword_4FBB3B0);
    if ( v3 ? *v3 : LODWORD(qword_4FBB3B0[2]) )
      return sub_1F493F0(a1);
  }
  if ( (unsigned __int8)sub_17006E0(a1[26]) )
  {
    v8 = (_QWORD *)sub_2105850();
    sub_1F46490((__int64)a1, v8, 1, 1, 0);
  }
  v6 = (_QWORD *)sub_1EABE60();
  sub_1F46490((__int64)a1, v6, 1, 1, 0);
  sub_1700880(&v9, a1[26]);
  v7 = sub_14A4230(&v9);
  sub_1F46490((__int64)a1, v7, 1, 1, 1u);
  if ( v10 )
    v10(&v9, &v9, 3);
  (*(void (__fastcall **)(__int64 *))(*a1 + 160))(a1);
  (*(void (__fastcall **)(__int64 *))(*a1 + 168))(a1);
  sub_1F474B0((__int64)a1);
  (*(void (__fastcall **)(__int64 *))(*a1 + 176))(a1);
  return sub_1F47EC0(a1);
}
