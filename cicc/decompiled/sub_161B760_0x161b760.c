// Function: sub_161B760
// Address: 0x161b760
//
void __fastcall sub_161B760(unsigned __int64 *src, unsigned __int64 *a2)
{
  __int64 v2; // rcx
  unsigned __int64 *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 112 )
  {
    sub_161B240(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 4;
    v3 = &src[v2];
    v4 = (8 * v2) >> 3;
    sub_161B760(src);
    sub_161B760(v3);
    sub_161B620((char *)src, (char *)v3, (__int64)a2, v4, a2 - v3);
  }
}
