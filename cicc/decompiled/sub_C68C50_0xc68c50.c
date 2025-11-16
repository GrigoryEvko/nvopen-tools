// Function: sub_C68C50
// Address: 0xc68c50
//
__int64 __fastcall sub_C68C50(__int64 a1, __int64 a2)
{
  _QWORD v3[4]; // [rsp+0h] [rbp-90h] BYREF
  char v4; // [rsp+20h] [rbp-70h]
  _QWORD v5[2]; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v6[2]; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v7[2]; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v8[2]; // [rsp+58h] [rbp-38h] BYREF
  _QWORD v9[5]; // [rsp+68h] [rbp-28h] BYREF

  v3[0] = "[{0}:{1}, byte={2}]: {3}";
  v3[2] = v9;
  v6[1] = a1 + 24;
  v3[3] = 4;
  v5[0] = &unk_49DC940;
  v5[1] = a1 + 8;
  v8[1] = a1 + 16;
  v4 = 1;
  v6[0] = &unk_49DC910;
  v7[0] = &unk_49DC910;
  v8[0] = &unk_49DC910;
  v9[0] = v8;
  v9[1] = v7;
  v9[2] = v6;
  v3[1] = 24;
  v7[1] = a1 + 20;
  v9[3] = v5;
  return sub_CB6840(a2, v3);
}
