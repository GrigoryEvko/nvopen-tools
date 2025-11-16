// Function: sub_38ECE00
// Address: 0x38ece00
//
__int64 __fastcall sub_38ECE00(__int64 a1, int a2, __int64 *a3, __int64 *a4)
{
  int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int8 v10; // [rsp+Fh] [rbp-51h]
  const char *v11; // [rsp+10h] [rbp-50h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  v6 = a2;
  v10 = sub_38ECD60(a1, a3, a4);
  if ( v10 )
    return 1;
  if ( a2 )
  {
    while ( !(unsigned __int8)sub_38EB510(a1, 1u, a3, (__int64)a4) )
    {
      if ( v6 == 1 )
        return v10;
      v7 = sub_3909460(a1);
      v8 = sub_39092B0(v7);
      v13 = 1;
      *a4 = v8;
      v11 = "expected ')' in parentheses expression";
      v12 = 3;
      if ( (unsigned __int8)sub_3909E20(a1, 18, &v11) )
        break;
      --v6;
    }
    return 1;
  }
  return v10;
}
