// Function: sub_1CFAC70
// Address: 0x1cfac70
//
__int64 __fastcall sub_1CFAC70(
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
  __int64 result; // rax
  __int64 *v12; // rax
  __int64 v13; // r14
  __int64 *v14; // r13
  __int64 v15; // r15
  int v16; // esi
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r15
  _QWORD *v22; // rax
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // r14
  unsigned int v26; // [rsp+Ch] [rbp-74h]
  _BYTE v27[16]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v28; // [rsp+20h] [rbp-60h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h]
  __int64 v31; // [rsp+40h] [rbp-40h]
  __int64 v32; // [rsp+48h] [rbp-38h]

  result = sub_1648720(*(_QWORD *)(a1 + 312));
  if ( !(_DWORD)result )
  {
    result = **(_QWORD **)(a2 - 48);
    if ( *(_BYTE *)(result + 8) == 16 )
    {
      result = **(_QWORD **)(result + 16);
      if ( *(_DWORD *)(result + 8) > 0x1FFu )
        return result;
    }
    else if ( *(_DWORD *)(result + 8) > 0x1FFu )
    {
      return result;
    }
    v12 = (__int64 *)sub_15F2050(a2);
    v13 = *(_QWORD *)a2;
    v14 = v12;
    v26 = (*(unsigned __int16 *)(a2 + 18) >> 2) & 7 | (*(unsigned __int16 *)(a2 + 18) << 11) & 0xFF0000;
    v15 = sub_1643350((_QWORD *)*v12);
    result = sub_1643360((_QWORD *)*v14);
    if ( *(_BYTE *)(v13 + 8) != 11 )
      return result;
    if ( result == v13 )
    {
      v16 = 4041;
    }
    else
    {
      v16 = 4040;
      if ( v15 != v13 )
        return result;
    }
    v29 = **(_QWORD **)(a2 - 48);
    v17 = **(_QWORD **)(a2 - 24);
    v31 = result;
    v30 = v17;
    v18 = sub_15E26F0(v14, v16, &v29, 3);
    v29 = sub_15A0680(v15, v26, 0);
    v19 = *(_QWORD *)(a2 - 48);
    v28 = 257;
    v20 = *(_QWORD *)(a1 + 304);
    v30 = v19;
    v31 = *(_QWORD *)(a2 - 24);
    v32 = *(_QWORD *)(v20 + 24 * (1LL - (*(_DWORD *)(v20 + 20) & 0xFFFFFFF)));
    v21 = *(_QWORD *)(*(_QWORD *)v18 + 24LL);
    v22 = sub_1648AB0(72, 5u, 0);
    v25 = (__int64)v22;
    if ( v22 )
    {
      sub_15F1EA0((__int64)v22, **(_QWORD **)(v21 + 16), 54, (__int64)(v22 - 15), 5, a2);
      *(_QWORD *)(v25 + 56) = 0;
      sub_15F5B40(v25, v21, v18, &v29, 4, (__int64)v27, 0, 0);
    }
    sub_164D160(a2, v25, a3, a4, a5, a6, v23, v24, a9, a10);
    return sub_15F20C0((_QWORD *)a2);
  }
  return result;
}
