// Function: sub_1BBBBC0
// Address: 0x1bbbbc0
//
void __fastcall sub_1BBBBC0(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)src <= 112 )
  {
    sub_1BBB410(src, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)src) >> 4;
    v5 = &src[v4];
    v6 = (8 * v4) >> 3;
    sub_1BBBBC0(src);
    sub_1BBBBC0(v5);
    sub_1BBB980((char *)src, (char *)v5, (__int64)a2, v6, a2 - v5, a3);
  }
}
