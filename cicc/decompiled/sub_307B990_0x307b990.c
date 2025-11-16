// Function: sub_307B990
// Address: 0x307b990
//
unsigned __int64 __fastcall sub_307B990(signed int a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // [rsp-38h] [rbp-38h] BYREF
  __int64 v9; // [rsp-30h] [rbp-30h]

  if ( a1 >= 0 )
    return 0;
  v8 = sub_2FF6F50(a3, a1, a2);
  v9 = v5;
  if ( sub_CA1930(&v8) == 1 )
    return 0x100000000LL;
  v6 = sub_2FF6F50(a3, a1, a2);
  v9 = v7;
  v8 = v6;
  result = (unsigned __int64)sub_CA1930(&v8) >> 5;
  if ( (int)result <= 0 )
    LODWORD(result) = 1;
  return (unsigned int)result;
}
