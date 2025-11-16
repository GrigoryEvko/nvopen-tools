// Function: sub_305C2E0
// Address: 0x305c2e0
//
__int64 __fastcall sub_305C2E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        unsigned int a8)
{
  __int64 v9; // r11
  __int64 v10; // rsi
  _QWORD v12[2]; // [rsp+0h] [rbp-30h] BYREF
  char v13; // [rsp+10h] [rbp-20h]
  __int64 v14; // [rsp+18h] [rbp-18h]
  __int64 v15; // [rsp+20h] [rbp-10h]

  v9 = *(_QWORD *)(a1 + 32);
  v10 = *(_QWORD *)(a1 + 16);
  v12[0] = a3;
  v12[1] = a4;
  v13 = a6;
  v15 = a5;
  v14 = a7;
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 1288LL))(
    v9,
    v10,
    v12,
    a2,
    a8,
    0);
  return 0;
}
