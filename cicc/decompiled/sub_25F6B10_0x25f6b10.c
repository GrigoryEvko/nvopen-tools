// Function: sub_25F6B10
// Address: 0x25f6b10
//
void __fastcall sub_25F6B10(unsigned int *src, unsigned int *a2, char *a3)
{
  char *v5; // r14
  unsigned int *v6; // r15
  unsigned int *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = &a3[(char *)a2 - (char *)src];
  v10 = (char *)a2 - (char *)src;
  v11 = a2 - src;
  if ( (char *)a2 - (char *)src <= 24 )
  {
    sub_25F6A60(src, a2);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 7;
      sub_25F6A60(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 24 );
    sub_25F6A60(v6, a2);
    if ( v10 > 28 )
    {
      v8 = 7;
      do
      {
        sub_25F5D80((char *)src, (char *)a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_25F5D80(a3, v5, (char *)src, v9);
      }
      while ( v11 > v8 );
    }
  }
}
