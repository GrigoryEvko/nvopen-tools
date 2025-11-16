// Function: sub_23043A0
// Address: 0x23043a0
//
__int64 *__fastcall sub_23043A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  bool v9; // zf
  __int64 v11; // [rsp+8h] [rbp-108h]
  _QWORD v12[8]; // [rsp+10h] [rbp-100h] BYREF
  char v13; // [rsp+54h] [rbp-BCh]

  sub_D053C0((__int64)v12, a2 + 8, a3, a4);
  v4 = v12[0];
  v5 = v12[1];
  v6 = v12[2];
  v11 = v12[3];
  v7 = v12[4];
  v8 = sub_22077B0(0xD0u);
  if ( v8 )
  {
    *(_QWORD *)(v8 + 8) = v4;
    *(_QWORD *)(v8 + 16) = v5;
    *(_QWORD *)(v8 + 24) = v6;
    *(_QWORD *)v8 = &unk_4A159A8;
    *(_QWORD *)(v8 + 32) = v11;
    *(_QWORD *)(v8 + 40) = v7;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)(v8 + 56) = v8 + 80;
    *(_QWORD *)(v8 + 64) = 16;
    *(_DWORD *)(v8 + 72) = 0;
    *(_BYTE *)(v8 + 76) = 1;
  }
  v9 = v13 == 0;
  *a1 = v8;
  if ( v9 )
    _libc_free(v12[6]);
  return a1;
}
