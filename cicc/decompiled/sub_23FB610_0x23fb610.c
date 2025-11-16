// Function: sub_23FB610
// Address: 0x23fb610
//
void __fastcall sub_23FB610(__int64 ***src, __int64 ***a2, char *a3)
{
  __int64 ***v5; // r14
  __int64 ***v6; // rdi
  __int64 v7; // r14
  __int64 v8; // rcx
  __int64 v9; // [rsp+8h] [rbp-48h]
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int64 ***v11; // [rsp+18h] [rbp-38h]

  v9 = (char *)a2 - (char *)src;
  v10 = a2 - src;
  v11 = (__int64 ***)&a3[(char *)a2 - (char *)src];
  if ( (char *)a2 - (char *)src <= 48 )
  {
    sub_23FB520(src, a2);
  }
  else
  {
    v5 = src;
    do
    {
      v6 = v5;
      v5 += 7;
      sub_23FB520(v6, v5);
    }
    while ( (char *)a2 - (char *)v5 > 48 );
    sub_23FB520(v5, a2);
    if ( v9 > 56 )
    {
      v7 = 7;
      do
      {
        sub_23FB470(src, a2, a3, v7);
        v8 = 2 * v7;
        v7 *= 4;
        sub_23FB470((__int64 ***)a3, v11, (char *)src, v8);
      }
      while ( v10 > v7 );
    }
  }
}
