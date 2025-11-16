// Function: sub_ECE300
// Address: 0xece300
//
__int64 __fastcall sub_ECE300(__int64 a1, __int64 (__fastcall *a2)(__int64), __int64 a3, char a4)
{
  __int64 result; // rax
  const char *v7; // [rsp+0h] [rbp-60h] BYREF
  char v8; // [rsp+20h] [rbp-40h]
  char v9; // [rsp+21h] [rbp-3Fh]

  if ( (unsigned __int8)sub_ECE2A0(a1, 9) )
    return 0;
  while ( 1 )
  {
    result = a2(a3);
    if ( (_BYTE)result )
      break;
    if ( (unsigned __int8)sub_ECE2A0(a1, 9) )
      return 0;
    if ( a4 )
    {
      v9 = 1;
      v7 = "unexpected token";
      v8 = 3;
      result = sub_ECE210(a1, 26, (__int64)&v7);
      if ( (_BYTE)result )
        break;
    }
  }
  return result;
}
