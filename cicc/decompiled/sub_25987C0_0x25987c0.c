// Function: sub_25987C0
// Address: 0x25987c0
//
_BOOL8 __fastcall sub_25987C0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rdi
  char v10; // dl
  char v11; // al
  __int64 v13; // [rsp-10h] [rbp-30h]
  unsigned __int64 v14[4]; // [rsp+0h] [rbp-20h] BYREF

  v4 = sub_250C680((__int64 *)(a1 + 72));
  sub_250D230(v14, v4, 6, 0);
  v5 = v14[0];
  v6 = sub_25294B0(a2, v14[0], v14[1], a1, 0, 0, 1);
  if ( v6 )
  {
    v7 = v6;
    v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL);
    if ( v8 == sub_2534F20 )
      v9 = v7 + 88;
    else
      v9 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v8)(v7, v5, v13);
    v10 = *(_BYTE *)(a1 + 97);
    v11 = *(_BYTE *)(a1 + 96) | v10 & *(_BYTE *)(v9 + 9);
    *(_BYTE *)(a1 + 97) = v11;
    return v10 == v11;
  }
  else
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
}
