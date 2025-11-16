// Function: sub_335E4B0
// Address: 0x335e4b0
//
void __fastcall sub_335E4B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _BYTE v6[8]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+8h] [rbp-28h]

  sub_335E470((__int64)v6, (__int64 *)a2, a1);
  while ( v7 )
  {
    ++*(_WORD *)(a2 + 250);
    sub_335E3B0((__int64)v6, a2, v2, v3, v4, v5);
  }
}
