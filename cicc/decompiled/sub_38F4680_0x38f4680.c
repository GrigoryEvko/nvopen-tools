// Function: sub_38F4680
// Address: 0x38f4680
//
__int64 __fastcall sub_38F4680(__int64 a1, char a2)
{
  unsigned int v4; // r12d
  _BOOL8 v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // ecx
  unsigned __int8 v9; // al
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v16[2]; // [rsp+10h] [rbp-A0h] BYREF
  const char *v17; // [rsp+20h] [rbp-90h] BYREF
  char v18; // [rsp+30h] [rbp-80h]
  char v19; // [rsp+31h] [rbp-7Fh]
  const char *v20; // [rsp+40h] [rbp-70h] BYREF
  char v21; // [rsp+50h] [rbp-60h]
  char v22; // [rsp+51h] [rbp-5Fh]
  _QWORD v23[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v24; // [rsp+70h] [rbp-40h]

  v15 = 0;
  v4 = sub_38EB9C0(a1, &v15);
  if ( !(_BYTE)v4 && v15 != 255 )
  {
    v19 = 1;
    v5 = 1;
    v16[0] = 0;
    v16[1] = 0;
    v17 = "unsupported encoding.";
    v18 = 3;
    if ( (v15 & 0xFFFFFFFFFFFFFF00LL) == 0 && ((unsigned int)(v15 & 7) - 2 <= 2 || (v15 & 7) == 0) )
      v5 = (v15 & 0x60) != 0;
    if ( (unsigned __int8)sub_3909CB0(a1, v5, &v17) )
      return 1;
    v22 = 1;
    v21 = 3;
    v20 = "unexpected token in directive";
    if ( (unsigned __int8)sub_3909E20(a1, 25, &v20) )
      return 1;
    v24 = 259;
    v23[0] = "expected identifier in directive";
    v9 = sub_38F0EE0(a1, v16, v6, v7);
    v4 = sub_3909CB0(a1, v9, v23);
    if ( (_BYTE)v4 )
    {
      return 1;
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 320);
      v23[0] = v16;
      v24 = 261;
      v11 = sub_38BF510(v10, (__int64)v23);
      v12 = *(__int64 **)(a1 + 328);
      v13 = v11;
      v14 = *v12;
      if ( a2 )
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v14 + 752))(v12, v13, (unsigned int)v15);
      else
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v14 + 760))(v12, v13, (unsigned int)v15);
    }
  }
  return v4;
}
