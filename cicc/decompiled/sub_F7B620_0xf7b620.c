// Function: sub_F7B620
// Address: 0xf7b620
//
void __fastcall sub_F7B620(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 *v7; // r14
  __int64 *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  __int64 *v13; // [rsp+18h] [rbp-38h]

  v11 = (char *)a2 - (char *)a1;
  v12 = ((char *)a2 - (char *)a1) >> 4;
  v13 = (__int64 *)((char *)a3 + (char *)a2 - (char *)a1);
  if ( (char *)a2 - (char *)a1 <= 96 )
  {
    sub_F7B350(a1, a2, a4);
  }
  else
  {
    v7 = a1;
    do
    {
      v8 = v7;
      v7 += 14;
      sub_F7B350(v8, v7, a4);
    }
    while ( (char *)a2 - (char *)v7 > 96 );
    sub_F7B350(v7, a2, a4);
    if ( v11 > 112 )
    {
      v9 = 7;
      do
      {
        sub_F7AF60(a1, a2, (__int64)a3, v9, a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_F7AF60(a3, v13, (__int64)a1, v10, a4);
      }
      while ( v12 > v9 );
    }
  }
}
