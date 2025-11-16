// Function: sub_95B9E0
// Address: 0x95b9e0
//
int __fastcall sub_95B9E0(const void *a1, size_t a2, const void *a3, size_t a4)
{
  size_t v7; // rdx
  int result; // eax
  __int64 v9; // rbx

  v7 = a4;
  if ( a2 <= a4 )
    v7 = a2;
  if ( !v7 || (result = memcmp(a1, a3, v7)) == 0 )
  {
    v9 = a2 - a4;
    result = 0x7FFFFFFF;
    if ( v9 <= 0x7FFFFFFF )
    {
      result = 0x80000000;
      if ( v9 >= (__int64)0xFFFFFFFF80000000LL )
        return v9;
    }
  }
  return result;
}
