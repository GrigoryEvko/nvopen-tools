// Function: sub_38E31C0
// Address: 0x38e31c0
//
__int64 __fastcall sub_38E31C0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _BOOL8 v5; // rsi
  _QWORD v6[3]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v7; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v8[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v9; // [rsp+30h] [rbp-60h]
  _QWORD v10[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v11; // [rsp+50h] [rbp-40h]
  const char *v12; // [rsp+60h] [rbp-30h] BYREF
  char v13; // [rsp+70h] [rbp-20h]
  char v14; // [rsp+71h] [rbp-1Fh]

  v6[0] = a3;
  v6[1] = a4;
  v7 = 0;
  if ( (unsigned __int8)sub_3909470(a1, &v7) )
    return 1;
  v9 = 1283;
  v8[0] = "expected function id in '";
  v8[1] = v6;
  v10[0] = v8;
  v11 = 770;
  v10[1] = "' directive";
  if ( (unsigned __int8)sub_3909D40(a1, a2, v10) )
    return 1;
  v12 = "expected function id within range [0, UINT_MAX)";
  v5 = *a2 > 0xFFFFFFFE;
  v14 = 1;
  v13 = 3;
  return sub_3909C80(a1, v5, v7, &v12);
}
