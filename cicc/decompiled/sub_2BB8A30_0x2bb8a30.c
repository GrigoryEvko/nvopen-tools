// Function: sub_2BB8A30
// Address: 0x2bb8a30
//
void __fastcall sub_2BB8A30(
        unsigned __int64 *a1,
        unsigned int *a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int *v9; // r14
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  v13 = (char *)a2 - (char *)a1;
  v14 = ((char *)a2 - (char *)a1) >> 6;
  v15 = (__int64)a3 + (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 384 )
  {
    sub_2BB7F80((__int64)a1, a2, a4, a4, a5, a6);
  }
  else
  {
    v9 = (unsigned int *)a1;
    do
    {
      v10 = (__int64)v9;
      v9 += 112;
      sub_2BB7F80(v10, v9, a4, a4, a5, a6);
    }
    while ( (char *)a2 - (char *)v9 > 384 );
    sub_2BB7F80((__int64)v9, a2, a4, a4, a5, a6);
    if ( v13 > 448 )
    {
      v11 = 7;
      do
      {
        sub_2BB8970(a1, (__int64)a2, (__int64)a3, v11, a4);
        v12 = 2 * v11;
        v11 *= 4;
        sub_2BB8970(a3, v15, (__int64)a1, v12, a4);
      }
      while ( v14 > v11 );
    }
  }
}
