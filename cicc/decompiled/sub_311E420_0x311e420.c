// Function: sub_311E420
// Address: 0x311e420
//
void __fastcall sub_311E420(unsigned __int64 **src, unsigned __int64 **a2, char *a3, __int64 a4)
{
  unsigned __int64 **v7; // r14
  unsigned __int64 **v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  unsigned __int64 **v13; // [rsp+18h] [rbp-38h]

  v11 = (char *)a2 - (char *)src;
  v12 = a2 - src;
  v13 = (unsigned __int64 **)&a3[(char *)a2 - (char *)src];
  if ( (char *)a2 - (char *)src <= 48 )
  {
    sub_311E330(src, a2, a4);
  }
  else
  {
    v7 = src;
    do
    {
      v8 = v7;
      v7 += 7;
      sub_311E330(v8, v7, a4);
    }
    while ( (char *)a2 - (char *)v7 > 48 );
    sub_311E330(v7, a2, a4);
    if ( v11 > 56 )
    {
      v9 = 7;
      do
      {
        sub_311E280(src, a2, a3, v9, a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_311E280((unsigned __int64 **)a3, v13, (char *)src, v10, a4);
      }
      while ( v12 > v9 );
    }
  }
}
