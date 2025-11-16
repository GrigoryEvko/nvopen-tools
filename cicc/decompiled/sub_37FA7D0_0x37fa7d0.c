// Function: sub_37FA7D0
// Address: 0x37fa7d0
//
_QWORD *__fastcall sub_37FA7D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD v17[2]; // [rsp+10h] [rbp-210h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-200h] BYREF
  _QWORD v19[8]; // [rsp+30h] [rbp-1F0h] BYREF
  _QWORD v20[4]; // [rsp+70h] [rbp-1B0h] BYREF
  char v21; // [rsp+90h] [rbp-190h]
  _QWORD v22[2]; // [rsp+98h] [rbp-188h] BYREF
  _QWORD v23[2]; // [rsp+A8h] [rbp-178h] BYREF
  _QWORD v24[3]; // [rsp+B8h] [rbp-168h] BYREF
  char *v25[3]; // [rsp+D0h] [rbp-150h] BYREF
  _BYTE v26[312]; // [rsp+E8h] [rbp-138h] BYREF

  v7 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 40LL))(
         *(_QWORD *)(a2 + 8),
         *(unsigned int *)(a4 + 2));
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(unsigned int *)(a4 + 10);
  v17[0] = v7;
  v17[1] = v10;
  v18[0] = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 40LL))(v8, v9);
  v20[0] = "{0} {1}";
  v20[2] = v24;
  v18[1] = v11;
  v22[1] = v18;
  v22[0] = &unk_49DB108;
  v23[0] = &unk_49DB108;
  v23[1] = v17;
  v24[0] = v23;
  v24[1] = v22;
  v19[5] = 0x100000000LL;
  v21 = 1;
  v19[0] = &unk_49DD288;
  v20[1] = 7;
  v20[3] = 2;
  v25[0] = v26;
  v25[1] = 0;
  v25[2] = (char *)256;
  v19[1] = 2;
  memset(&v19[2], 0, 24);
  v19[6] = v25;
  sub_CB5980((__int64)v19, 0, 0, 0);
  sub_CB6840((__int64)v19, (__int64)v20);
  v19[0] = &unk_49DD388;
  sub_CB5840((__int64)v19);
  sub_37FA2C0(a2 + 24, v25, v12, v13, v14, v15);
  if ( v25[0] != v26 )
    _libc_free((unsigned __int64)v25[0]);
  *a1 = 1;
  return a1;
}
