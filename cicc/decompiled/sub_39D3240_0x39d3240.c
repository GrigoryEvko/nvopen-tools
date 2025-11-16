// Function: sub_39D3240
// Address: 0x39d3240
//
void *__fastcall sub_39D3240(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *result; // rax
  __int64 v7; // rsi
  _QWORD *v8; // rdi
  __int64 v9; // rdx
  _QWORD v10[2]; // [rsp+0h] [rbp-C0h] BYREF
  void (__fastcall *v11)(_QWORD *, _QWORD *, __int64); // [rsp+10h] [rbp-B0h]
  void (__fastcall *v12)(_QWORD *, __int64 *); // [rsp+18h] [rbp-A8h]
  _QWORD v13[2]; // [rsp+20h] [rbp-A0h] BYREF
  void (__fastcall *v14)(_QWORD *, _QWORD *, __int64); // [rsp+30h] [rbp-90h]
  void (__fastcall *v15)(_QWORD *, __int64 *); // [rsp+38h] [rbp-88h]
  __int64 v16[4]; // [rsp+40h] [rbp-80h] BYREF
  int v17; // [rsp+60h] [rbp-60h]
  __int64 v18; // [rsp+68h] [rbp-58h]
  __int64 v19[4]; // [rsp+70h] [rbp-50h] BYREF
  int v20; // [rsp+90h] [rbp-30h]
  __int64 v21; // [rsp+98h] [rbp-28h]

  *(_BYTE *)a3 = *(_BYTE *)(a4 + 37);
  *(_BYTE *)(a3 + 1) = *(_BYTE *)(a4 + 38);
  *(_BYTE *)(a3 + 2) = *(_BYTE *)(a4 + 39);
  *(_BYTE *)(a3 + 3) = *(_BYTE *)(a4 + 40);
  *(_QWORD *)(a3 + 8) = *(_QWORD *)(a4 + 48);
  *(_DWORD *)(a3 + 16) = *(_DWORD *)(a4 + 56);
  *(_DWORD *)(a3 + 20) = *(_DWORD *)(a4 + 60);
  *(_BYTE *)(a3 + 24) = *(_BYTE *)(a4 + 64);
  *(_BYTE *)(a3 + 25) = *(_BYTE *)(a4 + 65);
  *(_DWORD *)(a3 + 80) = *(_DWORD *)(a4 + 76);
  *(_BYTE *)(a3 + 84) = *(_BYTE *)(a4 + 653);
  *(_BYTE *)(a3 + 85) = *(_BYTE *)(a4 + 655);
  *(_BYTE *)(a3 + 86) = *(_BYTE *)(a4 + 656);
  result = *(void **)(a4 + 640);
  *(_DWORD *)(a3 + 88) = (_DWORD)result;
  v7 = *(_QWORD *)(a4 + 664);
  if ( v7 )
  {
    v17 = 1;
    v8 = v10;
    v16[0] = (__int64)&unk_49EFBE0;
    memset(&v16[1], 0, 24);
    v18 = a3 + 96;
    sub_1DD5B60(v10, v7);
    if ( !v11 )
      goto LABEL_12;
    v12(v10, v16);
    if ( v11 )
      v11(v10, v10, 3);
    result = sub_16E7BC0(v16);
  }
  v7 = *(_QWORD *)(a4 + 672);
  if ( !v7 )
    return result;
  v20 = 1;
  v8 = v13;
  memset(&v19[1], 0, 24);
  v19[0] = (__int64)&unk_49EFBE0;
  v21 = a3 + 144;
  sub_1DD5B60(v13, v7);
  if ( !v14 )
LABEL_12:
    sub_4263D6(v8, v7, v9);
  v15(v13, v19);
  if ( v14 )
    v14(v13, v13, 3);
  return sub_16E7BC0(v19);
}
