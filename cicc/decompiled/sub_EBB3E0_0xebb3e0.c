// Function: sub_EBB3E0
// Address: 0xebb3e0
//
__int64 __fastcall sub_EBB3E0(__int64 a1, int a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax
  const char *v4; // [rsp+0h] [rbp-80h] BYREF
  size_t v5; // [rsp+8h] [rbp-78h]
  const char *v6; // [rsp+10h] [rbp-70h] BYREF
  char v7; // [rsp+30h] [rbp-50h]
  char v8; // [rsp+31h] [rbp-4Fh]
  const char *v9; // [rsp+40h] [rbp-40h] BYREF
  char v10; // [rsp+60h] [rbp-20h]
  char v11; // [rsp+61h] [rbp-1Fh]

  v4 = 0;
  v5 = 0;
  v8 = 1;
  v6 = "expected identifier";
  v7 = 3;
  v2 = sub_EB61F0(a1, (__int64 *)&v4);
  if ( (unsigned __int8)sub_ECE0A0(a1, v2, &v6) )
    return 1;
  v11 = 1;
  v10 = 3;
  v9 = "expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, &v9) )
    return 1;
  result = sub_EA98B0(a1, v4, v5, a2);
  if ( (_BYTE)result )
    return 1;
  return result;
}
