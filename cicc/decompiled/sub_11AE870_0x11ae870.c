// Function: sub_11AE870
// Address: 0x11ae870
//
__int64 __fastcall sub_11AE870(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v3; // ebx
  unsigned int v4; // r12d
  __int64 v6; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-28h]

  v2 = *(_QWORD *)(a2 + 8);
  v3 = sub_BCB060(v2);
  if ( !v3 )
    v3 = sub_AE43A0(a1[5].m128i_i64[1], v2);
  v7 = v3;
  if ( v3 > 0x40 )
  {
    sub_C43690((__int64)&v6, 0, 0);
    v9 = v3;
    sub_C43690((__int64)&v8, 0, 0);
  }
  else
  {
    v6 = 0;
    v9 = v3;
    v8 = 0;
  }
  v4 = sub_11AE3E0(a1, a2, (__int64)&v6);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return v4;
}
