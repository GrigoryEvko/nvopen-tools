// Function: sub_1920880
// Address: 0x1920880
//
void __fastcall sub_1920880(char *a1, char *a2, char *a3)
{
  __int64 v4; // rsi
  char *v7; // r15
  __int64 v8; // r14
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = a2 - a1;
  v7 = &a3[v4];
  if ( v4 <= 144 )
  {
    sub_1920210((__int64)a1, (__int64)a2);
  }
  else
  {
    v8 = (__int64)a1;
    do
    {
      v9 = v8;
      v8 += 168;
      sub_1920210(v9, v8);
    }
    while ( (__int64)&a2[-v8] > 144 );
    sub_1920210(v8, (__int64)a2);
    if ( v4 > 168 )
    {
      v10 = 7;
      do
      {
        sub_19207B0(a1, a2, a3, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_19207B0(a3, v7, a1, v11);
      }
      while ( (__int64)(0xAAAAAAAAAAAAAAABLL * (v4 >> 3)) > v10 );
    }
  }
}
