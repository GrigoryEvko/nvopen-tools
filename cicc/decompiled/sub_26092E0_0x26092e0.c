// Function: sub_26092E0
// Address: 0x26092e0
//
__int64 __fastcall sub_26092E0(unsigned int *src, unsigned int *a2, char *a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned int *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (a2 - src + 1) / 2;
  v10 = 4 * v6;
  v7 = &src[v6];
  if ( v6 <= a4 )
  {
    sub_25F6B10(src, &src[v6], a3);
    sub_25F6B10(v7, a2, a3);
  }
  else
  {
    sub_26092E0(src);
    sub_26092E0(v7);
  }
  sub_2608EC0((char *)src, (char *)v7, (__int64)a2, v10 >> 2, a2 - v7, a3, a4);
  return v9;
}
