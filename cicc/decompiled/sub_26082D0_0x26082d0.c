// Function: sub_26082D0
// Address: 0x26082d0
//
void __fastcall sub_26082D0(unsigned int *a1, unsigned int *a2)
{
  __int64 v2; // rbx

  if ( (char *)a2 - (char *)a1 <= 2128 )
  {
    sub_25F98B0(a1, a2);
  }
  else
  {
    v2 = 38 * ((__int64)(0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)a1) >> 3)) >> 1);
    sub_26082D0(a1, &a1[v2]);
    sub_26082D0(&a1[v2], a2);
    sub_2608140(
      (char *)a1,
      (__int64 *)&a1[v2],
      (__int64)a2,
      0x86BCA1AF286BCA1BLL * ((v2 * 4) >> 3),
      0x86BCA1AF286BCA1BLL * (((char *)a2 - (char *)&a1[v2]) >> 3));
  }
}
