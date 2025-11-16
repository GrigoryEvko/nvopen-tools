// Function: sub_305B5A0
// Address: 0x305b5a0
//
void __fastcall sub_305B5A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdi
  int v4; // edx
  _QWORD *v5; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v6; // [rsp-88h] [rbp-88h]
  _QWORD v7[4]; // [rsp-78h] [rbp-78h] BYREF
  char v8; // [rsp-58h] [rbp-58h]
  _QWORD v9[2]; // [rsp-50h] [rbp-50h] BYREF
  _QWORD v10[2]; // [rsp-40h] [rbp-40h] BYREF
  void *v11; // [rsp-30h] [rbp-30h] BYREF
  int v12; // [rsp-28h] [rbp-28h]
  _QWORD v13[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( *(_DWORD *)(a1 + 344) <= 0x59u || *(_DWORD *)(a1 + 336) <= 0x4Du )
  {
    v13[3] = v2;
    v3 = a1 + 336;
    v4 = *(_DWORD *)(v3 + 4);
    v7[0] = "NVPTX SM architecture \"{}\" and PTX version \"{}\" do not support {}. Requires SM >= 90 and PTX >= 78.";
    v7[2] = v13;
    v12 = v4;
    v13[0] = &v11;
    v9[0] = &unk_4A307F8;
    v13[1] = v10;
    v9[1] = a2;
    v10[0] = &unk_49DC910;
    v10[1] = v3;
    v13[2] = v9;
    v7[1] = 99;
    v7[3] = 3;
    v8 = 1;
    v11 = &unk_49E65E8;
    v6 = 263;
    v5 = v7;
    sub_C64D30((__int64)&v5, 1u);
  }
}
