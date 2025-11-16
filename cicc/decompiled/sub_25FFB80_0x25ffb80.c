// Function: sub_25FFB80
// Address: 0x25ffb80
//
__int64 __fastcall sub_25FFB80(unsigned int *a1, unsigned int *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]

  v5 = (__int64)(0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)a1) >> 3) + 1) / 2;
  v6 = 152 * v5;
  v7 = &a1[38 * v5];
  if ( v5 <= a4 )
  {
    sub_25F9DD0(a1, &a1[38 * v5], a3);
    sub_25F9DD0(v7, a2, a3);
  }
  else
  {
    sub_25FFB80(a1, &a1[38 * v5], a3);
    sub_25FFB80(v7, a2, a3);
  }
  sub_25FF3B0(
    (char *)a1,
    v7,
    (__int64)a2,
    0x86BCA1AF286BCA1BLL * (v6 >> 3),
    0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)v7) >> 3),
    a3,
    a4);
  return v9;
}
