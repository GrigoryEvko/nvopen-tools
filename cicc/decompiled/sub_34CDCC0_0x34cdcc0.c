// Function: sub_34CDCC0
// Address: 0x34cdcc0
//
__int64 __fastcall sub_34CDCC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // r11
  __int64 v11; // rsi
  _QWORD v13[2]; // [rsp+0h] [rbp-30h] BYREF
  char v14; // [rsp+10h] [rbp-20h]
  __int64 v15; // [rsp+18h] [rbp-18h]
  __int64 v16; // [rsp+20h] [rbp-10h]

  v10 = *(_QWORD *)(a1 + 32);
  v11 = *(_QWORD *)(a1 + 16);
  v13[0] = a3;
  v13[1] = a4;
  v14 = a5;
  v15 = a6;
  v16 = a9;
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, _QWORD, __int64))(*(_QWORD *)v10 + 1288LL))(
           v10,
           v11,
           v13,
           a2,
           a7,
           a8);
}
