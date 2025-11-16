// Function: sub_169D3F0
// Address: 0x169d3f0
//
char __fastcall sub_169D3F0(__int64 a1, double a2)
{
  char result; // al
  double v3; // [rsp+0h] [rbp-10h] BYREF
  unsigned int v4; // [rsp+8h] [rbp-8h]

  v4 = 64;
  v3 = a2;
  result = sub_169CFC0(a1, &unk_42AE9D0, (__int64 *)&v3);
  if ( v4 > 0x40 && v3 != 0.0 )
    return j_j___libc_free_0_0(*(_QWORD *)&v3);
  return result;
}
