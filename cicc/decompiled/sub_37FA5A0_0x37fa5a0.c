// Function: sub_37FA5A0
// Address: 0x37fa5a0
//
_QWORD *__fastcall sub_37FA5A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD v21[2]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v22[2]; // [rsp+20h] [rbp-220h] BYREF
  _QWORD v23[2]; // [rsp+30h] [rbp-210h] BYREF
  _QWORD v24[8]; // [rsp+40h] [rbp-200h] BYREF
  _QWORD v25[4]; // [rsp+80h] [rbp-1C0h] BYREF
  char v26; // [rsp+A0h] [rbp-1A0h]
  _QWORD v27[2]; // [rsp+A8h] [rbp-198h] BYREF
  _QWORD v28[2]; // [rsp+B8h] [rbp-188h] BYREF
  _QWORD v29[2]; // [rsp+C8h] [rbp-178h] BYREF
  _QWORD v30[3]; // [rsp+D8h] [rbp-168h] BYREF
  char *v31[3]; // [rsp+F0h] [rbp-150h] BYREF
  _BYTE v32[312]; // [rsp+108h] [rbp-138h] BYREF

  v7 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a2 + 8) + 40LL))(
         *(_QWORD *)(a2 + 8),
         *(unsigned int *)(a4 + 2));
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(unsigned int *)(a4 + 6);
  v21[0] = v7;
  v21[1] = v10;
  v11 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 40LL))(v8, v9);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(unsigned int *)(a4 + 18);
  v22[0] = v11;
  v22[1] = v14;
  v23[0] = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 40LL))(v12, v13);
  v25[0] = "{0} {1}::{2}";
  v25[2] = v30;
  v23[1] = v15;
  v27[1] = v23;
  v27[0] = &unk_49DB108;
  v28[0] = &unk_49DB108;
  v29[0] = &unk_49DB108;
  v29[1] = v21;
  v30[0] = v29;
  v30[1] = v28;
  v30[2] = v27;
  v24[5] = 0x100000000LL;
  v28[1] = v22;
  v24[0] = &unk_49DD288;
  v25[1] = 12;
  v25[3] = 3;
  v26 = 1;
  v31[0] = v32;
  v31[1] = 0;
  v31[2] = (char *)256;
  v24[1] = 2;
  memset(&v24[2], 0, 24);
  v24[6] = v31;
  sub_CB5980((__int64)v24, 0, 0, 0);
  sub_CB6840((__int64)v24, (__int64)v25);
  v24[0] = &unk_49DD388;
  sub_CB5840((__int64)v24);
  sub_37FA2C0(a2 + 24, v31, v16, v17, v18, v19);
  if ( v31[0] != v32 )
    _libc_free((unsigned __int64)v31[0]);
  *a1 = 1;
  return a1;
}
