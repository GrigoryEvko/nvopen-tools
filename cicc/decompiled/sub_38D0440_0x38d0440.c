// Function: sub_38D0440
// Address: 0x38d0440
//
__int64 __fastcall sub_38D0440(_QWORD *a1, __int64 a2)
{
  __int64 v3; // [rsp+8h] [rbp-8h] BYREF

  if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
    sub_38D0300(a1, a2, 1, &v3);
  else
    sub_38D01D0((__int64)a1, a2, 1, &v3);
  return v3;
}
