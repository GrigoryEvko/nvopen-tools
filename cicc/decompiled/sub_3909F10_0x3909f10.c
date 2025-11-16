// Function: sub_3909F10
// Address: 0x3909f10
//
__int64 __fastcall sub_3909F10(unsigned int *a1, __int64 (__fastcall *a2)(__int64), __int64 a3, char a4)
{
  __int64 result; // rax
  const char *v7; // [rsp+0h] [rbp-50h] BYREF
  char v8; // [rsp+10h] [rbp-40h]
  char v9; // [rsp+11h] [rbp-3Fh]

  if ( (unsigned __int8)sub_3909EB0(a1, 9) )
    return 0;
  while ( 1 )
  {
    result = a2(a3);
    if ( (_BYTE)result )
      break;
    if ( (unsigned __int8)sub_3909EB0(a1, 9) )
      return 0;
    if ( a4 )
    {
      v9 = 1;
      v7 = "unexpected token";
      v8 = 3;
      result = sub_3909E20(a1, 25, (__int64)&v7);
      if ( (_BYTE)result )
        break;
    }
  }
  return result;
}
