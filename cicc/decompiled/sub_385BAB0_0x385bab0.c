// Function: sub_385BAB0
// Address: 0x385bab0
//
void __fastcall sub_385BAB0(unsigned int *src, char *a2, char *a3, _QWORD *a4)
{
  char *v7; // r14
  unsigned int *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  unsigned int *v13; // [rsp+18h] [rbp-38h]

  v11 = a2 - (char *)src;
  v12 = (a2 - (char *)src) >> 2;
  v13 = (unsigned int *)&a3[a2 - (char *)src];
  if ( a2 - (char *)src <= 24 )
  {
    sub_385B9D0(src, a2, a4);
  }
  else
  {
    v7 = (char *)src;
    do
    {
      v8 = (unsigned int *)v7;
      v7 += 28;
      sub_385B9D0(v8, v7, a4);
    }
    while ( a2 - v7 > 24 );
    sub_385B9D0((unsigned int *)v7, a2, a4);
    if ( v11 > 28 )
    {
      v9 = 7;
      do
      {
        sub_385B560(src, (unsigned int *)a2, a3, v9, a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_385B560((unsigned int *)a3, v13, (char *)src, v10, a4);
      }
      while ( v12 > v9 );
    }
  }
}
