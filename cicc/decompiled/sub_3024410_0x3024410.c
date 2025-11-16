// Function: sub_3024410
// Address: 0x3024410
//
void __fastcall sub_3024410(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 *v4; // rdi
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  _QWORD v7[4]; // [rsp+0h] [rbp-140h] BYREF
  __int16 v8; // [rsp+20h] [rbp-120h]
  _QWORD v9[6]; // [rsp+30h] [rbp-110h] BYREF
  unsigned __int64 *v10; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v11[3]; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE v12[184]; // [rsp+88h] [rbp-B8h] BYREF

  v2 = *(_QWORD *)(a1 + 200);
  v9[5] = 0x100000000LL;
  v11[0] = (unsigned __int64)v12;
  v3 = v2 + 1288;
  v11[1] = 0;
  v9[0] = &unk_49DD288;
  v10 = v11;
  v11[2] = 128;
  v9[1] = 2;
  memset(&v9[2], 0, 24);
  sub_CB5980((__int64)v9, 0, 0, 0);
  sub_3023CE0(a1, a2, (__int64)v9, v3);
  v4 = *(__int64 **)(a1 + 224);
  v5 = v10[1];
  v6 = *v10;
  v8 = 261;
  v7[0] = v6;
  v7[1] = v5;
  sub_E99A90(v4, (__int64)v7);
  v9[0] = &unk_49DD388;
  sub_CB5840((__int64)v9);
  if ( (_BYTE *)v11[0] != v12 )
    _libc_free(v11[0]);
}
