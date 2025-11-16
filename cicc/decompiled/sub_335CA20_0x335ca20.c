// Function: sub_335CA20
// Address: 0x335ca20
//
void __fastcall sub_335CA20(char *src, char *a2, char *a3)
{
  char *v5; // r14
  char *v6; // r15
  char *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = &a3[a2 - src];
  v10 = a2 - src;
  v11 = (a2 - src) >> 3;
  if ( a2 - src <= 48 )
  {
    sub_335C960(src, a2);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 56;
      sub_335C960(v7, v6);
    }
    while ( a2 - v6 > 48 );
    sub_335C960(v6, a2);
    if ( v10 > 56 )
    {
      v8 = 7;
      do
      {
        sub_335BEF0(src, a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_335BEF0(a3, v5, src, v9);
      }
      while ( v11 > v8 );
    }
  }
}
