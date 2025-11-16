// Function: sub_C93CC0
// Address: 0xc93cc0
//
char __fastcall sub_C93CC0(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4)
{
  char result; // al
  __int64 v5; // [rsp+0h] [rbp-10h] BYREF
  __int64 v6; // [rsp+8h] [rbp-8h]

  v5 = a1;
  v6 = a2;
  result = sub_C93C00(&v5, a3, a4);
  if ( !result )
    return v6 != 0;
  return result;
}
