// Function: sub_E99C00
// Address: 0xe99c00
//
void *__fastcall sub_E99C00(__int64 a1, unsigned int *a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 *v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rax
  void *result; // rax
  _QWORD v8[4]; // [rsp+0h] [rbp-130h] BYREF
  __int16 v9; // [rsp+20h] [rbp-110h]
  _QWORD v10[6]; // [rsp+30h] [rbp-100h] BYREF
  __int64 *v11; // [rsp+60h] [rbp-D0h]
  _QWORD v12[3]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v13[168]; // [rsp+88h] [rbp-A8h] BYREF

  v10[5] = 0x100000000LL;
  v11 = v12;
  v12[0] = v13;
  v10[0] = &unk_49DD288;
  v12[1] = 0;
  v12[2] = 128;
  v10[1] = 2;
  memset(&v10[2], 0, 24);
  sub_CB5980((__int64)v10, 0, 0, 0);
  sub_E7FAD0(a2, (__int64)v10, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 152LL), 0, v2, v3);
  v4 = *(__int64 **)(a1 + 8);
  v5 = v11[1];
  v6 = *v11;
  v9 = 261;
  v8[0] = v6;
  v8[1] = v5;
  sub_E99A90(v4, (__int64)v8);
  v10[0] = &unk_49DD388;
  result = sub_CB5840((__int64)v10);
  if ( (_BYTE *)v12[0] != v13 )
    return (void *)_libc_free(v12[0], v8);
  return result;
}
