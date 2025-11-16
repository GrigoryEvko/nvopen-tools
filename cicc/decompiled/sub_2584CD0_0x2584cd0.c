// Function: sub_2584CD0
// Address: 0x2584cd0
//
_BOOL8 __fastcall sub_2584CD0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rdi
  _BOOL8 result; // rax
  char v10; // al
  char v11; // dl
  unsigned __int64 v12[4]; // [rsp+0h] [rbp-20h] BYREF

  v4 = sub_250C680((__int64 *)(a1 + 72));
  if ( v4 && (sub_250D230(v12, v4, 6, 0), v5 = v12[0], (v6 = sub_252B790(a2, v12[0], v12[1], a1, 0, 0, 1)) != 0) )
  {
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 48LL);
    if ( v7 == sub_2534F50 )
      v8 = v6 + 88;
    else
      v8 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v7)(v6, v5);
    result = 1;
    if ( !*(_BYTE *)(v8 + 9) )
    {
      v10 = *(_BYTE *)(a1 + 96);
      v11 = *(_BYTE *)(a1 + 97);
      *(_BYTE *)(a1 + 97) = v10;
      return v11 == v10;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  return result;
}
