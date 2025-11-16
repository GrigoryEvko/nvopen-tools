// Function: sub_95CB50
// Address: 0x95cb50
//
__int64 __fastcall sub_95CB50(const void **a1, const void *a2, size_t a3)
{
  unsigned int v3; // r15d
  char *v4; // r13
  char *v5; // r14

  v3 = 0;
  v4 = (char *)a1[1];
  if ( (unsigned __int64)v4 >= a3 )
  {
    v5 = (char *)*a1;
    if ( !a3 || !memcmp(*a1, a2, a3) )
    {
      v3 = 1;
      *a1 = &v5[a3];
      a1[1] = &v4[-a3];
    }
  }
  return v3;
}
