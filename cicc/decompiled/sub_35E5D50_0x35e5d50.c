// Function: sub_35E5D50
// Address: 0x35e5d50
//
void __fastcall sub_35E5D50(__int64 a1, char *a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v7; // r15
  char *v8; // r14
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = (__int64)&a2[-a1];
  v7 = a3 + v4;
  if ( v4 <= 144 )
  {
    sub_35E5200(a1, a2);
  }
  else
  {
    v8 = (char *)a1;
    do
    {
      v9 = (__int64)v8;
      v8 += 168;
      sub_35E5200(v9, v8);
    }
    while ( a2 - v8 > 144 );
    sub_35E5200((__int64)v8, a2);
    if ( v4 > 168 )
    {
      v10 = 7;
      do
      {
        sub_35E5C90(a1, (__int64)a2, a3, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_35E5C90(a3, v7, a1, v11);
      }
      while ( (__int64)(0xAAAAAAAAAAAAAAABLL * (v4 >> 3)) > v10 );
    }
  }
}
