// Function: sub_26BA4C0
// Address: 0x26ba4c0
//
size_t __fastcall sub_26BA4C0(int *a1, size_t a2)
{
  size_t v2; // r12
  __int64 v4; // [rsp+0h] [rbp-D0h] BYREF
  int v5[48]; // [rsp+10h] [rbp-C0h] BYREF

  v2 = a2;
  if ( a1 )
  {
    sub_C7D030(v5);
    sub_C7D280(v5, a1, a2);
    sub_C7D290(v5, &v4);
    return v4;
  }
  return v2;
}
