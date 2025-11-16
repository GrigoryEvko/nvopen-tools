// Function: sub_1CFAAD0
// Address: 0x1cfaad0
//
__int64 __fastcall sub_1CFAAD0(
        __int64 a1,
        __int64 a2,
        int a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rbx
  _QWORD *v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r13
  _BYTE v24[16]; // [rsp+0h] [rbp-70h] BYREF
  __int16 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h]
  __int64 v29; // [rsp+38h] [rbp-38h]

  v13 = (__int64 *)sub_15F2050(a2);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v14 = *(_QWORD *)(a2 - 8);
  else
    v14 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v26 = **(_QWORD **)(v14 + 24);
  v27 = **(_QWORD **)(v14 + 48);
  v25 = 257;
  v15 = sub_15E26F0(v13, a3, &v26, 2);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v16 = *(__int64 **)(a2 - 8);
  else
    v16 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v26 = *v16;
  v27 = v16[3];
  v17 = *(_QWORD *)(a1 + 304);
  v28 = v16[6];
  v29 = *(_QWORD *)(v17 + 24 * (1LL - (*(_DWORD *)(v17 + 20) & 0xFFFFFFF)));
  v18 = *(_QWORD *)(*(_QWORD *)v15 + 24LL);
  v19 = sub_1648AB0(72, 5u, 0);
  v22 = (__int64)v19;
  if ( v19 )
  {
    sub_15F1EA0((__int64)v19, **(_QWORD **)(v18 + 16), 54, (__int64)(v19 - 15), 5, a2);
    *(_QWORD *)(v22 + 56) = 0;
    sub_15F5B40(v22, v18, v15, &v26, 4, (__int64)v24, 0, 0);
  }
  sub_164D160(a2, v22, a4, a5, a6, a7, v20, v21, a10, a11);
  return sub_15F20C0((_QWORD *)a2);
}
