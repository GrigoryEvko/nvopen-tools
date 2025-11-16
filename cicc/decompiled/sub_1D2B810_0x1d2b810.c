// Function: sub_1D2B810
// Address: 0x1d2b810
//
__int64 __fastcall sub_1D2B810(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        int a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        unsigned __int16 a14,
        __int64 a15)
{
  unsigned __int8 *v18; // rax
  unsigned int v19; // ecx
  __int64 v20; // r8
  __int128 v21; // rax
  __int128 v23; // [rsp+0h] [rbp-70h]
  __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  int v26; // [rsp+38h] [rbp-38h]

  v18 = (unsigned __int8 *)(*(_QWORD *)(a8 + 40) + 16LL * (unsigned int)a9);
  v19 = *v18;
  v20 = *((_QWORD *)v18 + 1);
  v25 = 0;
  v26 = 0;
  *(_QWORD *)&v21 = sub_1D2B300(a1, 0x30u, (__int64)&v25, v19, v20, (__int64)&v25);
  if ( v25 )
  {
    v23 = v21;
    sub_161E7C0((__int64)&v25, v25);
    v21 = v23;
  }
  return sub_1D264C0(a1, 0, a2, a4, a5, a3, a7, a8, a9, v21, a10, a11, a12, a13, a6, a14, a15, 0);
}
