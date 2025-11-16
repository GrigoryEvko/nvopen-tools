// Function: sub_67B2C0
// Address: 0x67b2c0
//
__int64 __fastcall sub_67B2C0()
{
  __int64 v0; // rbp
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned int v6; // [rsp-5Ch] [rbp-5Ch]
  __int64 v7; // [rsp-58h] [rbp-58h] BYREF
  __int64 v8; // [rsp-50h] [rbp-50h]
  __int64 v9; // [rsp-48h] [rbp-48h]
  int v10; // [rsp-40h] [rbp-40h]
  _DWORD v11[2]; // [rsp-3Ch] [rbp-3Ch] BYREF
  int v12; // [rsp-34h] [rbp-34h] BYREF
  _DWORD v13[4]; // [rsp-30h] [rbp-30h] BYREF
  _QWORD v14[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( word_4F06418[0] != 27 )
    return 0;
  v14[3] = v0;
  v8 = 0x100000001LL;
  v7 = 0;
  v9 = 0;
  v10 = 0;
  v13[2] = dword_4F06650[0];
  v11[1] = 1;
  v13[1] = 1;
  sub_7BDB60(1);
  sub_866940(1, v11, v14, &v12, v13);
  sub_679930(0, 0, v2, v3);
  sub_67B070(&v7, 0, v4, v5);
  v6 = HIDWORD(v8);
  sub_679880((__int64)&v7);
  return v6;
}
