// Function: sub_23A20C0
// Address: 0x23a20c0
//
__int64 __fastcall sub_23A20C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r14
  char v7; // r13
  char v8; // bl
  _QWORD *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v16; // [rsp+0h] [rbp-D0h]
  bool v17; // [rsp+Fh] [rbp-C1h]
  _QWORD *v18; // [rsp+18h] [rbp-B8h] BYREF
  unsigned __int64 v19[22]; // [rsp+20h] [rbp-B0h] BYREF

  v6 = a3;
  v7 = a4;
  v8 = a5;
  v17 = *(_QWORD *)(a2 + 72) == *(_QWORD *)(a2 + 80);
  sub_2337A80((__int64)v19, a2, a3, a4, a5, a6);
  v9 = (_QWORD *)sub_22077B0(0x80u);
  if ( v9 )
  {
    v16 = v9;
    *v9 = &unk_4A0B4E8;
    sub_2337A80((__int64)(v9 + 1), (__int64)v19, (__int64)&unk_4A0B4E8, v10, v11, v12);
    v9 = v16;
  }
  *(_QWORD *)a1 = v9;
  *(_BYTE *)(a1 + 49) = v7;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = v6;
  *(_BYTE *)(a1 + 50) = v8;
  *(_BYTE *)(a1 + 51) = v17;
  v13 = (_QWORD *)sub_22077B0(0x10u);
  if ( v13 )
    *v13 = &unk_4A0B640;
  v18 = v13;
  sub_23A1F40((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)&v18);
  if ( v18 )
    (*(void (__fastcall **)(_QWORD *))(*v18 + 8LL))(v18);
  v14 = (_QWORD *)sub_22077B0(0x10u);
  if ( v14 )
    *v14 = &unk_4A0B680;
  v18 = v14;
  sub_23A1F40((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)&v18);
  if ( v18 )
    (*(void (__fastcall **)(_QWORD *))(*v18 + 8LL))(v18);
  sub_2337B30(v19);
  return a1;
}
