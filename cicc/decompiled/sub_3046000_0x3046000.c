// Function: sub_3046000
// Address: 0x3046000
//
__int64 __fastcall sub_3046000(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbp
  __int64 v4[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( (*(_BYTE *)(a2[1] + 864) & 1) != 0 )
    return 1;
  v4[1] = v2;
  v4[0] = sub_B2D7E0(*a2, "unsafe-fp-math", 0xEu);
  return sub_A72A30(v4);
}
