// Function: sub_9C88F0
// Address: 0x9c88f0
//
__int64 *__fastcall sub_9C88F0(__int64 *a1, __int64 a2, unsigned __int64 a3, _BYTE *a4, char a5)
{
  char v5; // al
  const char *v7; // [rsp+0h] [rbp-40h] BYREF
  char v8; // [rsp+20h] [rbp-20h]
  char v9; // [rsp+21h] [rbp-1Fh]

  if ( a3 > 0x21 )
  {
    v9 = 1;
    v7 = "Invalid alignment value";
    v8 = 3;
    sub_9C81F0(a1, a2 + 8, (__int64)&v7);
    return a1;
  }
  else
  {
    v5 = 0;
    if ( a3 )
    {
      a5 = a3 - 1;
      v5 = 1;
    }
    a4[1] = v5;
    *a4 = a5;
    *a1 = 1;
    return a1;
  }
}
