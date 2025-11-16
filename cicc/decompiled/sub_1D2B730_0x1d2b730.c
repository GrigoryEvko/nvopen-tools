// Function: sub_1D2B730
// Address: 0x1d2b730
//
__int64 __fastcall sub_1D2B730(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int64 a10,
        int a11,
        unsigned __int16 a12,
        __int64 a13,
        __int64 a14)
{
  unsigned __int8 *v16; // rax
  unsigned int v17; // ecx
  __int64 v18; // r8
  __int128 v19; // rax
  __int128 v21; // [rsp+0h] [rbp-70h]
  __int128 v22; // [rsp+20h] [rbp-50h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  int v24; // [rsp+38h] [rbp-38h]

  *(_QWORD *)&v22 = a5;
  *((_QWORD *)&v22 + 1) = a6;
  v16 = (unsigned __int8 *)(*(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8);
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  v23 = 0;
  v24 = 0;
  *(_QWORD *)&v19 = sub_1D2B300(a1, 0x30u, (__int64)&v23, v17, v18, a6);
  if ( v23 )
  {
    v21 = v19;
    sub_161E7C0((__int64)&v23, v23);
    v19 = v21;
  }
  return sub_1D264C0(a1, 0, 0, a2, a3, a4, v22, a7, a8, v19, a9, a10, a2, a3, a11, a12, a13, a14);
}
