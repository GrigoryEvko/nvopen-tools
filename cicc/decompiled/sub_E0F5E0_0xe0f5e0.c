// Function: sub_E0F5E0
// Address: 0xe0f5e0
//
__int64 __fastcall sub_E0F5E0(const void **a1, size_t a2, const void *a3)
{
  unsigned int v3; // r14d
  char *v4; // r13

  v3 = 0;
  v4 = (char *)*a1;
  if ( (_BYTE *)a1[1] - (_BYTE *)*a1 >= a2 && (!a2 || !memcmp(*a1, a3, a2)) )
  {
    v3 = 1;
    *a1 = &v4[a2];
  }
  return v3;
}
