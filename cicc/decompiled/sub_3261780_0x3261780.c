// Function: sub_3261780
// Address: 0x3261780
//
void __fastcall sub_3261780(unsigned int *src, unsigned int *a2, char *a3)
{
  unsigned int *v5; // r14
  unsigned int *v6; // r15
  unsigned int *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = (unsigned int *)&a3[(char *)a2 - (char *)src];
  v10 = (char *)a2 - (char *)src;
  v11 = ((char *)a2 - (char *)src) >> 4;
  if ( (char *)a2 - (char *)src <= 96 )
  {
    sub_32613F0(src, (__int64)a2);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 28;
      sub_32613F0(v7, (__int64)v6);
    }
    while ( (char *)a2 - (char *)v6 > 96 );
    sub_32613F0(v6, (__int64)a2);
    if ( v10 > 112 )
    {
      v8 = 7;
      do
      {
        sub_3260C60(src, a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_3260C60((unsigned int *)a3, v5, (char *)src, v9);
      }
      while ( v11 > v8 );
    }
  }
}
