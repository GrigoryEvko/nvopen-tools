// Function: sub_302AD70
// Address: 0x302ad70
//
void __fastcall sub_302AD70(__int64 a1)
{
  __int64 *v1; // rdi
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  _QWORD v4[4]; // [rsp+0h] [rbp-130h] BYREF
  __int16 v5; // [rsp+20h] [rbp-110h]
  _QWORD v6[6]; // [rsp+30h] [rbp-100h] BYREF
  unsigned __int64 *v7; // [rsp+60h] [rbp-D0h]
  unsigned __int64 v8[3]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v9[168]; // [rsp+88h] [rbp-A8h] BYREF

  v6[5] = 0x100000000LL;
  v7 = v8;
  v8[0] = (unsigned __int64)v9;
  v6[0] = &unk_49DD288;
  v8[1] = 0;
  v8[2] = 128;
  v6[1] = 2;
  memset(&v6[2], 0, 24);
  sub_CB5980((__int64)v6, 0, 0, 0);
  sub_302AC40(a1, **(_QWORD **)(a1 + 232), (__int64)v6);
  v1 = *(__int64 **)(a1 + 224);
  v2 = v7[1];
  v3 = *v7;
  v5 = 261;
  v4[0] = v3;
  v4[1] = v2;
  sub_E99A90(v1, (__int64)v4);
  v6[0] = &unk_49DD388;
  sub_CB5840((__int64)v6);
  if ( (_BYTE *)v8[0] != v9 )
    _libc_free(v8[0]);
}
