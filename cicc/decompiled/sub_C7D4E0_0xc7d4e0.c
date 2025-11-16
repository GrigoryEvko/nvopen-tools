// Function: sub_C7D4E0
// Address: 0xc7d4e0
//
__int64 __fastcall sub_C7D4E0(unsigned __int8 *a1, _QWORD *a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // dl
  unsigned __int8 v4; // cl
  char v5; // dl

  if ( a2[1] != 32 )
  {
    if ( a2[1] <= 0x20u && a2[2] <= 0x1Fu )
      sub_C8D290(a2, a2 + 3, 32, 1);
    a2[1] = 32;
  }
  for ( result = 0; result != 32; result += 2 )
  {
    v3 = *a1++;
    v4 = v3;
    v5 = a0123456789abcd_10[v3 & 0xF] | 0x20;
    *(_BYTE *)(*a2 + result) = a0123456789abcd_10[v4 >> 4] | 0x20;
    *(_BYTE *)(*a2 + result + 1) = v5;
  }
  return result;
}
