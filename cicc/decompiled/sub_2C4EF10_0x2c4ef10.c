// Function: sub_2C4EF10
// Address: 0x2c4ef10
//
__int64 __fastcall sub_2C4EF10(unsigned int *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 **a5, _QWORD *a6)
{
  __int64 v6; // rax
  unsigned int *v9; // r13
  __int64 v10; // rbx
  unsigned int *v11; // rdi
  __int64 v13; // [rsp-10h] [rbp-60h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v6 = ((char *)a2 - (char *)a1) >> 3;
  v9 = a1;
  v16 = 2 * a4;
  if ( 2 * a4 <= v6 )
  {
    v15 = 16 * a4;
    v10 = -2 * a4;
    do
    {
      v11 = v9;
      v9 = (unsigned int *)((char *)v9 + v15);
      a3 = sub_2C4EBF0(v11, &v9[v10], &v9[v10], v9, a3, (__int64)a6, a5, a6);
      v6 = ((char *)a2 - (char *)v9) >> 3;
    }
    while ( v6 >= v16 );
  }
  if ( v6 > a4 )
    v6 = a4;
  sub_2C4EBF0(v9, &v9[2 * v6], &v9[2 * v6], a2, a3, (__int64)a6, a5, a6);
  return v13;
}
