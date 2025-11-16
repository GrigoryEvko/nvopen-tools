// Function: sub_2B0E610
// Address: 0x2b0e610
//
void __fastcall sub_2B0E610(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 *v3; // r14
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r11
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-38h]

  v3 = (__int64 *)((char *)a3 + (char *)a2 - (char *)a1);
  v11 = ((char *)a2 - (char *)a1) >> 4;
  if ( (char *)a2 - (char *)a1 <= 96 )
  {
    sub_2B0E570((__int64)a1, (__int64)a2);
  }
  else
  {
    v6 = (__int64)a1;
    do
    {
      v7 = v6;
      v6 += 112;
      sub_2B0E570(v7, v6);
    }
    while ( (__int64)a2 - v6 > 96 );
    sub_2B0E570(v6, (__int64)a2);
    if ( v8 > 112 )
    {
      v9 = 7;
      do
      {
        sub_2B0B1E0(a1, a2, (__int64)a3, v9);
        v10 = 2 * v9;
        v9 *= 4;
        sub_2B0B1E0(a3, v3, (__int64)a1, v10);
      }
      while ( v11 > v9 );
    }
  }
}
