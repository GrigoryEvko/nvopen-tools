// Function: sub_EA2660
// Address: 0xea2660
//
__int64 __fastcall sub_EA2660(__int64 a1, _QWORD *a2)
{
  _BOOL8 v3; // rsi
  __int64 v4; // [rsp+8h] [rbp-78h] BYREF
  const char *v5; // [rsp+10h] [rbp-70h] BYREF
  char v6; // [rsp+30h] [rbp-50h]
  char v7; // [rsp+31h] [rbp-4Fh]
  const char *v8; // [rsp+40h] [rbp-40h] BYREF
  char v9; // [rsp+60h] [rbp-20h]
  char v10; // [rsp+61h] [rbp-1Fh]

  v4 = 0;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v4) )
    return 1;
  v10 = 1;
  v9 = 3;
  v8 = "expected function id";
  if ( (unsigned __int8)sub_ECE130(a1, a2, &v8) )
    return 1;
  v5 = "expected function id within range [0, UINT_MAX)";
  v3 = *a2 > 0xFFFFFFFE;
  v7 = 1;
  v6 = 3;
  return sub_ECE070(a1, v3, v4, &v5);
}
