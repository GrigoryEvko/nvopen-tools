// Function: sub_39C5B00
// Address: 0x39c5b00
//
__int64 __fastcall sub_39C5B00(int *a1, __int64 **a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 result; // rax
  int v5[12]; // [rsp+Fh] [rbp-31h] BYREF

  v2 = *a2;
  v3 = a2[1];
  if ( v3 != *a2 )
  {
    do
    {
      LOBYTE(v5[0]) = v2[2];
      sub_16C1870(a1, v5, 1u);
      result = *v2;
      v2 = (__int64 *)(*v2 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (result & 4) != 0 )
        v2 = 0;
    }
    while ( v3 != v2 );
  }
  return result;
}
