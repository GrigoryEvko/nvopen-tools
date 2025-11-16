// Function: sub_2C4F9C0
// Address: 0x2c4f9c0
//
void __fastcall sub_2C4F9C0(unsigned int *a1, unsigned int *a2, unsigned int *a3, __int64 **a4, _BYTE **a5)
{
  unsigned int *v8; // rbx
  unsigned int *v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // [rsp+0h] [rbp-50h]
  __int64 v13; // [rsp+8h] [rbp-48h]
  unsigned int *v14; // [rsp+10h] [rbp-40h]

  v12 = (char *)a2 - (char *)a1;
  v13 = ((char *)a2 - (char *)a1) >> 3;
  v14 = (unsigned int *)((char *)a3 + (char *)a2 - (char *)a1);
  if ( (char *)a2 - (char *)a1 <= 48 )
  {
    sub_2C4F5D0(a1, a2, a4, a5);
  }
  else
  {
    v8 = a1;
    do
    {
      v9 = v8;
      v8 += 14;
      sub_2C4F5D0(v9, v8, a4, a5);
    }
    while ( (char *)a2 - (char *)v8 > 48 );
    sub_2C4F5D0(v8, a2, a4, a5);
    if ( v12 > 56 )
    {
      v10 = 7;
      do
      {
        sub_2C4EB30(a1, a2, (__int64)a3, v10, a4, a5);
        v11 = 2 * v10;
        v10 *= 4;
        sub_2C4EB30(a3, v14, (__int64)a1, v11, a4, a5);
      }
      while ( v13 > v10 );
    }
  }
}
