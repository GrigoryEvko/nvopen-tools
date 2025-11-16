// Function: sub_257EED0
// Address: 0x257eed0
//
_BOOL8 __fastcall sub_257EED0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rdi
  char v9; // dl
  char v10; // al
  unsigned __int64 v12[4]; // [rsp+0h] [rbp-20h] BYREF

  v4 = sub_250C680((__int64 *)(a1 + 72));
  if ( v4 && (sub_250D230(v12, v4, 6, 0), v5 = v12[0], (v6 = sub_251BBC0(a2, v12[0], v12[1], a1, 0, 0, 1)) != 0) )
  {
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL);
    if ( v7 == sub_2534F20 )
      v8 = v6 + 88;
    else
      v8 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v7)(v6, v5);
    v9 = *(_BYTE *)(a1 + 97);
    v10 = *(_BYTE *)(a1 + 96) | v9 & *(_BYTE *)(v8 + 9);
    *(_BYTE *)(a1 + 97) = v10;
    return v9 == v10;
  }
  else
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
}
