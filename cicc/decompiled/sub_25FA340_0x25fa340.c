// Function: sub_25FA340
// Address: 0x25fa340
//
void __fastcall sub_25FA340(unsigned int *src, unsigned int *a2)
{
  __int64 v2; // rcx
  unsigned int *v3; // r14
  __int64 v4; // rbx

  if ( (char *)a2 - (char *)src <= 56 )
  {
    sub_25F6A60(src, a2);
  }
  else
  {
    v2 = ((char *)a2 - (char *)src) >> 3;
    v3 = &src[v2];
    v4 = (4 * v2) >> 2;
    sub_25FA340(src);
    sub_25FA340(v3);
    sub_25FA200((char *)src, (char *)v3, (__int64)a2, v4, a2 - v3);
  }
}
