// Function: sub_1D0E0F0
// Address: 0x1d0e0f0
//
void __fastcall sub_1D0E0F0(__int64 a1, __int64 a2)
{
  _BYTE v2[8]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v3; // [rsp+8h] [rbp-28h]

  sub_1D0E0B0((__int64)v2, (__int64 *)a2, a1);
  while ( v3 )
  {
    ++*(_WORD *)(a2 + 224);
    sub_1D0DFF0((__int64)v2);
  }
}
