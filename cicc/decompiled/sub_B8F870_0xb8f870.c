// Function: sub_B8F870
// Address: 0xb8f870
//
void __fastcall sub_B8F870(int *a1, unsigned int *a2, int *a3)
{
  int *v3; // r14
  unsigned int *v6; // r15
  __int64 v7; // rdi
  __int64 v8; // r11
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-38h]

  v3 = (int *)((char *)a3 + (char *)a2 - (char *)a1);
  v11 = ((char *)a2 - (char *)a1) >> 4;
  if ( (char *)a2 - (char *)a1 <= 96 )
  {
    sub_B8F7D0((__int64)a1, a2);
  }
  else
  {
    v6 = (unsigned int *)a1;
    do
    {
      v7 = (__int64)v6;
      v6 += 28;
      sub_B8F7D0(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 96 );
    sub_B8F7D0((__int64)v6, a2);
    if ( v8 > 112 )
    {
      v9 = 7;
      do
      {
        sub_B8EAB0(a1, (int *)a2, a3, v9);
        v10 = 2 * v9;
        v9 *= 4;
        sub_B8EAB0(a3, v3, a1, v10);
      }
      while ( v11 > v9 );
    }
  }
}
