// Function: sub_1D2B660
// Address: 0x1d2b660
//
__int64 __fastcall sub_1D2B660(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int8 *v13; // rax
  unsigned int v14; // ecx
  __int64 v15; // r8
  __int128 v16; // rax
  _QWORD *v17; // r10
  __int128 v19; // [rsp-50h] [rbp-B0h]
  __int128 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]

  v13 = (unsigned __int8 *)(*(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v21 = 0;
  v22 = 0;
  *(_QWORD *)&v16 = sub_1D2B300(a1, 0x30u, (__int64)&v21, v14, v15, (__int64)&v21);
  v17 = a1;
  if ( v21 )
  {
    v20 = v16;
    sub_161E7C0((__int64)&v21, v21);
    v17 = a1;
    v16 = v20;
  }
  *((_QWORD *)&v19 + 1) = a6;
  *(_QWORD *)&v19 = a5;
  return sub_1D260A0(v17, 0, 0, a2, a3, a4, v19, a7, a8, v16, a2, a3, a9);
}
