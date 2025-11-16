// Function: sub_2F10960
// Address: 0x2f10960
//
void *__fastcall sub_2F10960(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *result; // rax
  __int64 v7; // rsi
  _QWORD *v8; // rdi
  __int64 v9; // rdx
  _QWORD v10[2]; // [rsp+0h] [rbp-F0h] BYREF
  void (__fastcall *v11)(_QWORD *, _QWORD *, __int64); // [rsp+10h] [rbp-E0h]
  void (__fastcall *v12)(_QWORD *, _QWORD *); // [rsp+18h] [rbp-D8h]
  _QWORD v13[2]; // [rsp+20h] [rbp-D0h] BYREF
  void (__fastcall *v14)(_QWORD *, _QWORD *, __int64); // [rsp+30h] [rbp-C0h]
  void (__fastcall *v15)(_QWORD *, _QWORD *); // [rsp+38h] [rbp-B8h]
  _QWORD v16[8]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v17[14]; // [rsp+80h] [rbp-70h] BYREF

  *(_BYTE *)a3 = *(_BYTE *)(a4 + 37);
  *(_BYTE *)(a3 + 1) = *(_BYTE *)(a4 + 38);
  *(_BYTE *)(a3 + 2) = *(_BYTE *)(a4 + 39);
  *(_BYTE *)(a3 + 3) = *(_BYTE *)(a4 + 40);
  *(_QWORD *)(a3 + 8) = *(_QWORD *)(a4 + 48);
  *(_DWORD *)(a3 + 16) = *(_QWORD *)(a4 + 56);
  *(_DWORD *)(a3 + 20) = 1LL << *(_BYTE *)(a4 + 64);
  *(_BYTE *)(a3 + 24) = *(_BYTE *)(a4 + 65);
  *(_BYTE *)(a3 + 25) = *(_BYTE *)(a4 + 66);
  *(_DWORD *)(a3 + 128) = *(_QWORD *)(a4 + 80);
  *(_DWORD *)(a3 + 132) = *(_DWORD *)(a4 + 88);
  *(_BYTE *)(a3 + 136) = *(_BYTE *)(a4 + 666);
  *(_BYTE *)(a3 + 137) = *(_BYTE *)(a4 + 668);
  *(_BYTE *)(a3 + 138) = *(_BYTE *)(a4 + 669);
  *(_BYTE *)(a3 + 139) = *(_BYTE *)(a4 + 670);
  *(_BYTE *)(a3 + 140) = *(_BYTE *)(a4 + 120);
  result = *(void **)(a4 + 656);
  *(_DWORD *)(a3 + 144) = (_DWORD)result;
  if ( *(_QWORD *)(a4 + 672) )
  {
    memset(&v16[1], 0, 32);
    v16[5] = 0x100000000LL;
    v16[0] = &unk_49DD210;
    v16[6] = a3 + 152;
    sub_CB5980((__int64)v16, 0, 0, 0);
    v7 = *(_QWORD *)(a4 + 672);
    v8 = v10;
    sub_2E31000(v10, v7);
    if ( !v11 )
      goto LABEL_12;
    v12(v10, v16);
    if ( v11 )
      v11(v10, v10, 3);
    v16[0] = &unk_49DD210;
    result = sub_CB5840((__int64)v16);
  }
  if ( !*(_QWORD *)(a4 + 680) )
    return result;
  v17[5] = 0x100000000LL;
  v17[0] = &unk_49DD210;
  v17[6] = a3 + 200;
  memset(&v17[1], 0, 32);
  sub_CB5980((__int64)v17, 0, 0, 0);
  v7 = *(_QWORD *)(a4 + 680);
  v8 = v13;
  sub_2E31000(v13, v7);
  if ( !v14 )
LABEL_12:
    sub_4263D6(v8, v7, v9);
  v15(v13, v17);
  if ( v14 )
    v14(v13, v13, 3);
  v17[0] = &unk_49DD210;
  return sub_CB5840((__int64)v17);
}
