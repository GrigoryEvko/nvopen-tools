// Function: sub_161DA50
// Address: 0x161da50
//
void __fastcall sub_161DA50(int *a1, unsigned int *a2, int *a3)
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
    sub_161D9B0((__int64)a1, a2);
  }
  else
  {
    v6 = (unsigned int *)a1;
    do
    {
      v7 = (__int64)v6;
      v6 += 28;
      sub_161D9B0(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 96 );
    sub_161D9B0((__int64)v6, a2);
    if ( v8 > 112 )
    {
      v9 = 7;
      do
      {
        sub_161D0F0(a1, (int *)a2, a3, v9);
        v10 = 2 * v9;
        v9 *= 4;
        sub_161D0F0(a3, v3, a1, v10);
      }
      while ( v11 > v9 );
    }
  }
}
