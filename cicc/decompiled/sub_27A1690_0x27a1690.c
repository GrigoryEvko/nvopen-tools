// Function: sub_27A1690
// Address: 0x27a1690
//
void __fastcall sub_27A1690(int *a1, int *a2, char *a3)
{
  int *v5; // r14
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = (int *)&a3[(char *)a2 - (char *)a1];
  v10 = (char *)a2 - (char *)a1;
  v11 = ((char *)a2 - (char *)a1) >> 5;
  if ( (char *)a2 - (char *)a1 <= 192 )
  {
    sub_27A1220((__int64)a1, (__int64)a2);
  }
  else
  {
    v6 = (__int64)a1;
    do
    {
      v7 = v6;
      v6 += 224;
      sub_27A1220(v7, v6);
    }
    while ( (__int64)a2 - v6 > 192 );
    sub_27A1220(v6, (__int64)a2);
    if ( v10 > 224 )
    {
      v8 = 7;
      do
      {
        sub_27A15E0(a1, a2, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_27A15E0((int *)a3, v5, (char *)a1, v9);
      }
      while ( v11 > v8 );
    }
  }
}
