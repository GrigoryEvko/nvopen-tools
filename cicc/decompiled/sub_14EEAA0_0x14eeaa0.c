// Function: sub_14EEAA0
// Address: 0x14eeaa0
//
__int64 *__fastcall sub_14EEAA0(__int64 *a1, __int64 a2, unsigned __int64 a3, int *a4)
{
  const char *v5; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+10h] [rbp-20h]
  char v7; // [rsp+11h] [rbp-1Fh]

  if ( a3 <= 0x1E )
  {
    *a4 = 1 << a3 >> 1;
    *a1 = 1;
    return a1;
  }
  else
  {
    v7 = 1;
    v5 = "Invalid alignment value";
    v6 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v5);
    return a1;
  }
}
