// Function: sub_155CE20
// Address: 0x155ce20
//
__int64 *__fastcall sub_155CE20(__int64 *a1, unsigned __int64 a2, char a3)
{
  char *v3; // r8
  unsigned __int64 v5; // rax
  char v6; // [rsp+14h] [rbp-1Ch] BYREF
  _BYTE v7[19]; // [rsp+15h] [rbp-1Bh] BYREF

  if ( a2 )
  {
    v3 = v7;
    do
    {
      *--v3 = a2 % 0xA + 48;
      v5 = a2;
      a2 /= 0xAu;
    }
    while ( v5 > 9 );
  }
  else
  {
    v6 = 48;
    v3 = &v6;
  }
  if ( a3 )
    *--v3 = 45;
  *a1 = (__int64)(a1 + 2);
  sub_155CB60(a1, v3, (__int64)v7);
  return a1;
}
