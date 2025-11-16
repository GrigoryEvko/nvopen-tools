// Function: sub_1CFA790
// Address: 0x1cfa790
//
char __fastcall sub_1CFA790(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rbx
  _QWORD *v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // r13
  _QWORD *v22; // [rsp+8h] [rbp-78h]
  _BYTE v23[16]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+30h] [rbp-50h] BYREF
  __int64 v26; // [rsp+38h] [rbp-48h]
  _QWORD *v27; // [rsp+40h] [rbp-40h]
  __int64 v28; // [rsp+48h] [rbp-38h]

  LODWORD(v10) = sub_1648720(*(_QWORD *)(a1 + 312));
  if ( (_DWORD)v10 == 1 )
  {
    LOBYTE(v10) = sub_15F32D0(a2);
    if ( !(_BYTE)v10 && (*(_BYTE *)(a2 + 18) & 1) == 0 )
    {
      v10 = **(_QWORD **)(a2 - 24);
      if ( *(_DWORD *)(v10 + 8) <= 0x1FFu )
      {
        v11 = (__int64 *)sub_15F2050(a2);
        v22 = *(_QWORD **)(a2 - 24);
        v25 = **(_QWORD **)(a2 - 48);
        v26 = *v22;
        v12 = sub_15E26F0(v11, 4047, &v25, 2);
        v13 = sub_1643360((_QWORD *)*v11);
        v25 = sub_159C470(v13, 0, 0);
        v14 = *(_QWORD *)(a2 - 48);
        v24 = 257;
        v15 = *(_QWORD *)(a1 + 304);
        v26 = v14;
        v27 = v22;
        v28 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
        v16 = *(_QWORD *)(*(_QWORD *)v12 + 24LL);
        v17 = sub_1648AB0(72, 5u, 0);
        v20 = (__int64)v17;
        if ( v17 )
        {
          sub_15F1EA0((__int64)v17, **(_QWORD **)(v16 + 16), 54, (__int64)(v17 - 15), 5, a2);
          *(_QWORD *)(v20 + 56) = 0;
          sub_15F5B40(v20, v16, v12, &v25, 4, (__int64)v23, 0, 0);
        }
        sub_164D160(a2, v20, a3, a4, a5, a6, v18, v19, a9, a10);
        LOBYTE(v10) = sub_15F20C0((_QWORD *)a2);
      }
    }
  }
  return v10;
}
