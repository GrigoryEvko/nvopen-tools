// Function: sub_1CFAE70
// Address: 0x1cfae70
//
__int64 __fastcall sub_1CFAE70(
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
  unsigned int v11; // ebx
  int v12; // edx
  __int64 result; // rax
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // r14
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // r13
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 *v27; // r15
  __int64 *v28; // r8
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r15
  _QWORD *v34; // rax
  int v35; // edx
  unsigned int v36; // ecx
  int v37; // edx
  __int64 v38; // [rsp-90h] [rbp-90h]
  __int64 v39[2]; // [rsp-88h] [rbp-88h] BYREF
  __int64 v40; // [rsp-78h] [rbp-78h]
  __int64 v41; // [rsp-68h] [rbp-68h] BYREF
  __int64 v42; // [rsp-60h] [rbp-60h]
  __int64 v43; // [rsp-58h] [rbp-58h]
  __int64 v44; // [rsp-50h] [rbp-50h]
  __int64 v45; // [rsp-48h] [rbp-48h]
  __int64 v46; // [rsp-40h] [rbp-40h]

  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) )
    BUG();
  v11 = *(_DWORD *)(v10 + 36);
  v12 = sub_1648720(*(_QWORD *)(a1 + 312));
  if ( v11 <= 0xE7A )
  {
    if ( v11 > 0xE78 )
    {
      result = 1;
    }
    else if ( v11 > 0xE6A )
    {
      result = v11 - 3697 < 6 ? 1 : -1;
    }
    else
    {
      result = 0;
      if ( v11 <= 0xE66 )
        result = v11 - 3669 < 4 ? 1 : -1;
    }
LABEL_7:
    if ( v12 != (_DWORD)result )
      return result;
    goto LABEL_16;
  }
  if ( v11 > 0xFDE )
  {
    result = 0xFFFFFFFFLL;
    if ( v11 == 4163 )
    {
      if ( v12 != 2 )
        return result;
      v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v41 = **(_QWORD **)(a2 + 24 * (1 - v14));
      v42 = **(_QWORD **)(a2 + 24 * (2 - v14));
      v15 = (__int64 *)sub_15F2050(a2);
      v16 = sub_15E26F0(v15, 4166, &v41, 2);
      v17 = *(_DWORD *)(a2 + 20);
      LOWORD(v40) = 257;
      v18 = v17 & 0xFFFFFFF;
      v41 = *(_QWORD *)(a2 - 24 * v18);
      v42 = *(_QWORD *)(a2 + 24 * (1 - v18));
      v43 = *(_QWORD *)(a2 + 24 * (2 - v18));
      v44 = *(_QWORD *)(a2 + 24 * (3 - v18));
      v19 = *(_QWORD *)(a1 + 304);
      v45 = *(_QWORD *)(a2 + 24 * (4 - v18));
      v46 = *(_QWORD *)(v19 + 24 * (1LL - (*(_DWORD *)(v19 + 20) & 0xFFFFFFF)));
      v20 = *(_QWORD *)(*(_QWORD *)v16 + 24LL);
      v21 = sub_1648AB0(72, 7u, 0);
      v24 = (__int64)v21;
      if ( v21 )
      {
        sub_15F1EA0((__int64)v21, **(_QWORD **)(v20 + 16), 54, (__int64)(v21 - 21), 7, a2);
        *(_QWORD *)(v24 + 56) = 0;
        sub_15F5B40(v24, v20, v16, &v41, 6, (__int64)v39, 0, 0);
      }
      goto LABEL_14;
    }
    goto LABEL_7;
  }
  result = (unsigned int)-(v11 < 0xFDC);
  if ( v12 != (_DWORD)result )
    return result;
LABEL_16:
  result = sub_1C30260(v11);
  if ( !(_BYTE)result )
  {
    if ( v11 - 4060 > 2 )
      return result;
    v27 = (__int64 *)sub_15F2050(a2);
    v28 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v41 = *(_QWORD *)a2;
    v38 = (__int64)v28;
    v42 = *v28;
    v29 = sub_15E26F0(v27, 4043, &v41, 2);
    v30 = sub_1643360((_QWORD *)*v27);
    v31 = sub_159C470(v30, 1, 0);
    LOWORD(v43) = 257;
    v32 = *(_QWORD *)(a1 + 304);
    v39[0] = v31;
    v39[1] = v38;
    v40 = *(_QWORD *)(v32 + 24 * (1LL - (*(_DWORD *)(v32 + 20) & 0xFFFFFFF)));
    v33 = *(_QWORD *)(*(_QWORD *)v29 + 24LL);
    v34 = sub_1648AB0(72, 4u, 0);
    v24 = (__int64)v34;
    if ( v34 )
    {
      sub_15F1EA0((__int64)v34, **(_QWORD **)(v33 + 16), 54, (__int64)(v34 - 12), 4, a2);
      *(_QWORD *)(v24 + 56) = 0;
      sub_15F5B40(v24, v33, v29, v39, 3, (__int64)&v41, 0, 0);
    }
LABEL_14:
    sub_164D160(a2, v24, a3, a4, a5, a6, v22, v23, a9, a10);
    return sub_15F20C0((_QWORD *)a2);
  }
  switch ( v11 )
  {
    case 0xE67u:
      v36 = 720896;
      v37 = 4038;
      return sub_1CFA920(a1, a2, v37, v36, a3, a4, a5, a6, v25, v26, a9, a10);
    case 0xE68u:
      v36 = 720896;
      v37 = 4039;
      return sub_1CFA920(a1, a2, v37, v36, a3, a4, a5, a6, v25, v26, a9, a10);
    case 0xE69u:
      v36 = 851968;
      v37 = 4040;
      return sub_1CFA920(a1, a2, v37, v36, a3, a4, a5, a6, v25, v26, a9, a10);
    case 0xE6Au:
      v36 = 786432;
      v37 = 4040;
      return sub_1CFA920(a1, a2, v37, v36, a3, a4, a5, a6, v25, v26, a9, a10);
    case 0xE72u:
      v35 = 4038;
      goto LABEL_24;
    case 0xE73u:
      v35 = 4039;
      goto LABEL_24;
    case 0xE75u:
      v35 = 4040;
      goto LABEL_24;
    case 0xE76u:
      v35 = 4041;
LABEL_24:
      result = sub_1CFAAD0(a1, a2, v35, a3, a4, a5, a6, v25, v26, a9, a10);
      break;
    default:
      return result;
  }
  return result;
}
