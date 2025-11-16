// Function: sub_F07460
// Address: 0xf07460
//
void __fastcall sub_F07460(__int64 *src, __int64 *a2, char *a3)
{
  __int64 *v5; // r14
  __int64 *v6; // r15
  __int64 *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = (__int64 *)&a3[(char *)a2 - (char *)src];
  v10 = (char *)a2 - (char *)src;
  v11 = a2 - src;
  if ( (char *)a2 - (char *)src <= 48 )
  {
    sub_F07380(src, a2);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 7;
      sub_F07380(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 48 );
    sub_F07380(v6, a2);
    if ( v10 > 56 )
    {
      v8 = 7;
      do
      {
        sub_F071A0(src, a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_F071A0((__int64 *)a3, v5, (char *)src, v9);
      }
      while ( v11 > v8 );
    }
  }
}
