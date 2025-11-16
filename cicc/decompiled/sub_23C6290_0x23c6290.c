// Function: sub_23C6290
// Address: 0x23c6290
//
int __fastcall sub_23C6290(void *s1, unsigned __int64 a2, const void *a3, unsigned __int64 a4, char a5)
{
  unsigned __int64 v6; // rdx
  int result; // eax
  unsigned __int64 v10; // rcx
  void *v12; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v13; // [rsp+18h] [rbp-38h]

  v6 = a2;
  if ( a4 <= a2 )
    v6 = a4;
  if ( a4 < a2 )
  {
    v12 = s1;
    v13 = v6;
    result = sub_C92E90(&v12, (__int64)a3, a4);
    v10 = a4;
    if ( result )
      return result;
    return 2 * (a2 == v10) - 1;
  }
  v12 = s1;
  v13 = a2;
  result = sub_C92E90(&v12, (__int64)a3, v6);
  if ( result )
    return result;
  if ( a4 != a2 )
  {
    v10 = a2;
    return 2 * (a2 == v10) - 1;
  }
  if ( a5 && a4 )
  {
    result = memcmp(s1, a3, a4);
    if ( result )
      return (result >> 31) | 1;
  }
  return result;
}
