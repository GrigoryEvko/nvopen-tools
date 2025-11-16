// Function: sub_25FB820
// Address: 0x25fb820
//
void __fastcall sub_25FB820(unsigned int *src, unsigned int *a2)
{
  __int64 v2; // rcx
  unsigned int *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 56 )
  {
    sub_25F6BE0(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 3;
    v3 = &src[v2];
    v4 = (4 * v2) >> 2;
    sub_25FB820(src);
    sub_25FB820(v3);
    sub_25FB6E0((char *)src, (char *)v3, (__int64)a2, v4, a2 - v3);
  }
}
