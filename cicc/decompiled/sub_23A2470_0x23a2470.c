// Function: sub_23A2470
// Address: 0x23a2470
//
void __fastcall sub_23A2470(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  _BOOL8 v4; // rsi
  char v5; // r14
  __int16 v6; // bx
  __int64 v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  __int16 v10; // [rsp-92h] [rbp-92h] BYREF
  __int64 v11; // [rsp-90h] [rbp-90h] BYREF
  __int64 v12[2]; // [rsp-88h] [rbp-88h] BYREF
  __int64 v13; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int64 v14[5]; // [rsp-70h] [rbp-70h] BYREF
  int v15; // [rsp-48h] [rbp-48h]

  if ( byte_4FDDBC8 )
  {
    v4 = 1;
    v5 = *(_BYTE *)(a1 + 32);
    if ( !byte_4FDD4C8 && HIDWORD(qword_5033EE0) == HIDWORD(a3) )
      v4 = (_DWORD)qword_5033EE0 != (_DWORD)a3;
    sub_28448C0(&v10, v4, 0);
    v6 = v10;
    v7 = sub_22077B0(0x10u);
    if ( v7 )
    {
      *(_WORD *)(v7 + 8) = v6;
      *(_QWORD *)v7 = &unk_4A124B8;
    }
    v13 = v7;
    v11 = 0;
    memset(v14, 0, sizeof(v14));
    v15 = 0;
    v8 = (_QWORD *)sub_22077B0(0x10u);
    if ( v8 )
      *v8 = &unk_4A0B640;
    v12[0] = (__int64)v8;
    sub_23A1F40(v14, (unsigned __int64 *)v12);
    sub_233EFE0(v12);
    v9 = (_QWORD *)sub_22077B0(0x10u);
    if ( v9 )
      *v9 = &unk_4A0B680;
    v12[0] = (__int64)v9;
    sub_23A1F40(v14, (unsigned __int64 *)v12);
    sub_233EFE0(v12);
    sub_233F7D0(&v11);
    sub_234CFF0((__int64)v12, &v13, v5);
    sub_23571D0(a2, v12);
    sub_233EFE0(v12);
    sub_233F7F0((__int64)v14);
    sub_233F7D0(&v13);
  }
}
