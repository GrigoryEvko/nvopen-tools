// Function: sub_2FBF290
// Address: 0x2fbf290
//
void __fastcall sub_2FBF290(char *src, char *a2, char *a3, __int64 a4)
{
  char *v7; // r14
  char *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  char *v13; // [rsp+18h] [rbp-38h]

  v11 = a2 - src;
  v12 = (a2 - src) >> 2;
  v13 = &a3[a2 - src];
  if ( a2 - src <= 24 )
  {
    sub_2FBF180(src, a2, a4);
  }
  else
  {
    v7 = src;
    do
    {
      v8 = v7;
      v7 += 28;
      sub_2FBF180(v8, v7, a4);
    }
    while ( a2 - v7 > 24 );
    sub_2FBF180(v7, a2, a4);
    if ( v11 > 28 )
    {
      v9 = 7;
      do
      {
        sub_2FBEED0(src, a2, a3, v9, a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_2FBEED0(a3, v13, src, v10, a4);
      }
      while ( v12 > v9 );
    }
  }
}
