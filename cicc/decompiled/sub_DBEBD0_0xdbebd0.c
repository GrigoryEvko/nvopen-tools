// Function: sub_DBEBD0
// Address: 0xdbebd0
//
__int64 __fastcall sub_DBEBD0(__int64 a1, __int64 a2)
{
  _BYTE v3[8]; // [rsp+0h] [rbp-10h] BYREF
  __int64 v4; // [rsp+8h] [rbp-8h]

  v3[0] = 0;
  v4 = a1;
  sub_DBE8E0(a2, (__int64)v3);
  return v3[0] ^ 1u;
}
