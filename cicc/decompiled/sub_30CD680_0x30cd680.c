// Function: sub_30CD680
// Address: 0x30cd680
//
void __fastcall sub_30CD680(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        unsigned __int8 *a4,
        unsigned __int8 *a5,
        _DWORD *a6,
        char a7,
        char *a8)
{
  __int64 v11; // rsi
  bool v12; // zf
  char v13; // r9
  char v14; // [rsp+18h] [rbp-58h]
  char v15; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v17[8]; // [rsp+30h] [rbp-40h] BYREF

  v11 = *a2;
  v17[1] = a6;
  v12 = *a6 == 0x80000000;
  v15 = a7;
  v17[0] = &v15;
  v13 = v12;
  v16 = v11;
  if ( v11 )
  {
    v14 = v12;
    sub_B96E90((__int64)&v16, v11, 1);
    v13 = v14;
  }
  sub_30CD350(a1, &v16, a3, a4, a5, v13, (void (__fastcall *)(__int64, _QWORD *))sub_30CD980, (__int64)v17, a8);
  if ( v16 )
    sub_B91220((__int64)&v16, v16);
}
