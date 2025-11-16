// Function: sub_1CFA920
// Address: 0x1cfa920
//
__int64 __fastcall sub_1CFA920(
        __int64 a1,
        __int64 a2,
        int a3,
        unsigned int a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 *v12; // r14
  __int64 v13; // r13
  __int64 **v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rbx
  _QWORD *v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r13
  _BYTE v26[16]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h]
  __int64 v31; // [rsp+48h] [rbp-38h]

  v12 = (__int64 *)sub_15F2050(a2);
  v13 = sub_1643350((_QWORD *)*v12);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v14 = *(__int64 ***)(a2 - 8);
  else
    v14 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v28 = **v14;
  v29 = *v14[3];
  v15 = sub_15E26F0(v12, a3, &v28, 2);
  v27 = 257;
  v28 = sub_15A0680(v13, a4, 0);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v16 = *(__int64 **)(a2 - 8);
  else
    v16 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v29 = *v16;
  v17 = *(_QWORD *)(a1 + 304);
  v30 = v16[3];
  v31 = *(_QWORD *)(v17 + 24 * (1LL - (*(_DWORD *)(v17 + 20) & 0xFFFFFFF)));
  v18 = *(_QWORD *)(*(_QWORD *)v15 + 24LL);
  v19 = sub_1648AB0(72, 5u, 0);
  v22 = (__int64)v19;
  if ( v19 )
  {
    sub_15F1EA0((__int64)v19, **(_QWORD **)(v18 + 16), 54, (__int64)(v19 - 15), 5, a2);
    *(_QWORD *)(v22 + 56) = 0;
    sub_15F5B40(v22, v18, v15, &v28, 4, (__int64)v26, 0, 0);
  }
  sub_164D160(a2, v22, a5, a6, a7, a8, v20, v21, a11, a12);
  return sub_15F20C0((_QWORD *)a2);
}
