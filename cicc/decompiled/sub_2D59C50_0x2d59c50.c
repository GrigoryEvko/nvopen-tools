// Function: sub_2D59C50
// Address: 0x2d59c50
//
bool __fastcall sub_2D59C50(char *a1, __int64 a2)
{
  bool result; // al
  _BYTE *v3; // [rsp-48h] [rbp-48h] BYREF
  __int64 v4; // [rsp-40h] [rbp-40h] BYREF
  char *v5; // [rsp-38h] [rbp-38h] BYREF
  bool v6; // [rsp-28h] [rbp-28h]

  if ( (unsigned __int8)*a1 <= 0x1Cu )
    return 0;
  v3 = 0;
  v4 = 0;
  if ( !(unsigned __int8)sub_2D57670(a1, &v3, &v4) || *v3 != 84 )
    return 0;
  sub_2D59A60((__int64)&v5, (__int64)v3, a2);
  result = v6;
  if ( v6 )
    return v5 == a1;
  return result;
}
