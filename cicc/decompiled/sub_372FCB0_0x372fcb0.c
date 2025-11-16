// Function: sub_372FCB0
// Address: 0x372fcb0
//
unsigned __int64 __fastcall sub_372FCB0(int *a1, unsigned __int64 a2)
{
  char v2; // al
  unsigned __int64 v3; // rbx
  int v5[8]; // [rsp+Fh] [rbp-21h] BYREF

  v2 = a2 & 0x7F;
  v3 = a2 >> 7;
  for ( LOBYTE(v5[0]) = a2 & 0x7F; v3; LOBYTE(v5[0]) = v2 )
  {
    LOBYTE(v5[0]) = v2 | 0x80;
    sub_C7D060(a1, v5, 1u);
    v2 = v3 & 0x7F;
    v3 >>= 7;
  }
  return sub_C7D060(a1, v5, 1u);
}
