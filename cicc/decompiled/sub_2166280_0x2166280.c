// Function: sub_2166280
// Address: 0x2166280
//
__int64 __fastcall sub_2166280(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8)
{
  __int64 v9; // r11
  __int64 v10; // rsi
  _QWORD v12[2]; // [rsp+0h] [rbp-20h] BYREF
  char v13; // [rsp+10h] [rbp-10h]
  __int64 v14; // [rsp+18h] [rbp-8h]

  v9 = *(_QWORD *)(a1 + 24);
  v10 = *(_QWORD *)(a1 + 8);
  v12[0] = a3;
  v12[1] = a4;
  v13 = a5;
  v14 = a6;
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, _QWORD, __int64))(*(_QWORD *)v9 + 736LL))(
           v9,
           v10,
           v12,
           a2,
           a7,
           a8);
}
