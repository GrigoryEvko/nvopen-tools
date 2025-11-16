// Function: sub_259B590
// Address: 0x259b590
//
__int64 __fastcall sub_259B590(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  char v6; // al
  char v8; // [rsp+7h] [rbp-59h] BYREF
  unsigned __int64 v9; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-50h] BYREF
  __m128i v11; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v12; // [rsp+30h] [rbp-30h]

  v9 = 0;
  v3 = sub_250D070((_QWORD *)(a1 + 72));
  if ( *(_BYTE *)v3 > 0x1Cu )
  {
    v4 = sub_B43CB0(v3);
    v9 = v4;
    v5 = v4;
    if ( *(_BYTE *)v3 == 22 )
      goto LABEL_3;
    if ( v4 )
      goto LABEL_4;
LABEL_7:
    *(_BYTE *)(a1 + 96) = *(_BYTE *)(a1 + 97);
    return 1;
  }
  if ( *(_BYTE *)v3 != 22 )
    goto LABEL_7;
LABEL_3:
  v5 = *(_QWORD *)(v3 + 24);
  v6 = *(_BYTE *)(v5 + 32);
  v9 = v5;
  if ( (v6 & 0xFu) - 7 > 1 )
    return 1;
LABEL_4:
  sub_250D230((unsigned __int64 *)&v11, v5, 4, 0);
  if ( (unsigned __int8)sub_259B4A0(a2, a1, &v11, 1, &v8, 0, 0) )
    return 1;
  v12 = &v9;
  v11.m128i_i64[0] = a2;
  v11.m128i_i64[1] = a1;
  v10[0] = a2;
  v10[1] = a1;
  if ( (unsigned __int8)sub_252FFB0(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2584B10,
                          (__int64)&v11,
                          a1,
                          v3,
                          1,
                          1,
                          1,
                          (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2537B10,
                          (__int64)v10) )
    return 1;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
