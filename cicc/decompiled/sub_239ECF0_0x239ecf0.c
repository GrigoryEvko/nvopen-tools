// Function: sub_239ECF0
// Address: 0x239ecf0
//
void __fastcall sub_239ECF0(const void **a1, const void *a2, size_t a3)
{
  char *v3; // r13
  char *v4; // r14

  v3 = (char *)a1[1];
  if ( (unsigned __int64)v3 >= a3 )
  {
    v4 = (char *)*a1;
    if ( !a3 || !memcmp(*a1, a2, a3) )
    {
      *a1 = &v4[a3];
      a1[1] = &v3[-a3];
    }
  }
}
