// Function: sub_1F493F0
// Address: 0x1f493f0
//
__int64 __fastcall sub_1F493F0(__int64 *a1)
{
  _DWORD *v1; // rax
  _QWORD *v2; // rsi
  _QWORD *v4; // rax
  __m128i *v5; // rsi
  _QWORD *v6; // rsi
  __m128i v7; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v8)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]

  v1 = sub_16D40F0((__int64)qword_4FBB3B0);
  if ( v1 )
  {
    if ( *v1 != 2 )
    {
LABEL_3:
      sub_1F49190((__int64)a1, (__int64)&unk_4FC9148, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FC5C8C, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FCAE51, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FCAACC, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FC434C, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FC7F6C, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FC8CB4, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FC3604, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FC84CC, 0, 0);
      sub_1F49190((__int64)a1, (__int64)&unk_4FCA74C, 0, 0);
      v2 = (_QWORD *)sub_1857160();
      sub_1F46490((__int64)a1, v2, 1, 1, 0);
      (*(void (__fastcall **)(__int64 *))(*a1 + 176))(a1);
      return sub_1F47EC0(a1);
    }
  }
  else if ( LODWORD(qword_4FBB3B0[2]) != 2 )
  {
    goto LABEL_3;
  }
  if ( (unsigned __int8)sub_17006E0(a1[26]) )
  {
    v6 = (_QWORD *)sub_2105850();
    sub_1F46490((__int64)a1, v6, 1, 1, 0);
  }
  v4 = (_QWORD *)sub_1EABE60();
  sub_1F46490((__int64)a1, v4, 1, 1, 0);
  sub_1700880(&v7, a1[26]);
  v5 = sub_14A4230(&v7);
  sub_1F46490((__int64)a1, v5, 1, 1, 1u);
  if ( v8 )
    v8(&v7, &v7, 3);
  (*(void (__fastcall **)(__int64 *))(*a1 + 160))(a1);
  (*(void (__fastcall **)(__int64 *))(*a1 + 168))(a1);
  sub_1F474B0((__int64)a1);
  return 1;
}
