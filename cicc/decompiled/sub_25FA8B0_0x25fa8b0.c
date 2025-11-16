// Function: sub_25FA8B0
// Address: 0x25fa8b0
//
void __fastcall sub_25FA8B0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  signed __int64 v2; // rbx

  if ( (char *)a2 - (char *)a1 <= 336 )
  {
    sub_25FA3C0(a1, a2);
  }
  else
  {
    v2 = ((0xAAAAAAAAAAAAAAABLL * (a2 - a1)) & 0xFFFFFFFFFFFFFFFELL)
       + ((__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - a1)) >> 1);
    sub_25FA8B0(a1, &a1[v2]);
    sub_25FA8B0(&a1[v2], a2);
    sub_25F92A0(
      (char *)a1,
      (char *)&a1[v2],
      (__int64)a2,
      0xAAAAAAAAAAAAAAABLL * ((v2 * 8) >> 3),
      0xAAAAAAAAAAAAAAABLL * (a2 - &a1[v2]));
  }
}
