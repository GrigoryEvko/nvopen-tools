// Function: sub_3959780
// Address: 0x3959780
//
__int64 __fastcall sub_3959780(__int64 a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rcx
  unsigned int v4; // r12d
  unsigned __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-28h]
  unsigned __int64 v8; // [rsp+10h] [rbp-20h]
  unsigned int v9; // [rsp+18h] [rbp-18h]

  sub_14C2530((__int64)&v6, a2, a1, 0, 0, 0, 0, 0);
  v2 = v7;
  if ( v7 > 0x40 )
    v3 = *(_QWORD *)(v6 + 8LL * ((v7 - 1) >> 6));
  else
    v3 = v6;
  v4 = ((v3 & (1LL << ((unsigned __int8)v7 - 1))) != 0) + 1;
  if ( v9 > 0x40 && v8 )
  {
    j_j___libc_free_0_0(v8);
    v2 = v7;
  }
  if ( v2 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return v4;
}
