// Function: sub_3876200
// Address: 0x3876200
//
__int64 __fastcall sub_3876200(
        __int64 *a1,
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
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 **v13; // r13
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 ***v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  __int64 v19; // r15
  __int64 ***v20; // r14
  __int64 v21; // rdi
  unsigned int v22; // ebx
  int v23; // eax
  __int64 v24; // rdi
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r8
  int v28; // r9d
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rcx
  unsigned __int64 v32; // rax
  __int64 ***v33; // rax

  v11 = *a1;
  v12 = sub_1456040(*(_QWORD *)(a2 + 40));
  v13 = (__int64 **)sub_1456E10(v11, v12);
  v16 = sub_38761C0(a1, *(_QWORD *)(a2 + 32), v13, a3, a4, a5, a6, v14, v15, a9, a10);
  v19 = *(_QWORD *)(a2 + 40);
  v20 = v16;
  if ( *(_WORD *)(v19 + 24) )
    goto LABEL_7;
  v21 = *(_QWORD *)(v19 + 32);
  v22 = *(_DWORD *)(v21 + 32);
  if ( v22 <= 0x40 )
  {
    v32 = *(_QWORD *)(v21 + 24);
    if ( v32 && (v32 & (v32 - 1)) == 0 )
    {
      _BitScanReverse64(&v32, v32);
      v25 = v22 + (v32 ^ 0x3F) - 64;
      goto LABEL_5;
    }
LABEL_7:
    v33 = sub_38761C0(a1, v19, v13, a3, a4, a5, a6, v17, v18, a9, a10);
    v29 = (__int64)v20;
    v30 = 17;
    v31 = (__int64)v33;
    return sub_3874770((__int64)a1, v30, v29, v31, v27, v28, *(double *)a3.m128_u64, a4, a5);
  }
  v23 = sub_16A5940(v21 + 24);
  v24 = v21 + 24;
  if ( v23 != 1 )
    goto LABEL_7;
  v25 = sub_16A57B0(v24);
LABEL_5:
  v26 = sub_15A0680((__int64)v13, v22 - 1 - v25, 0);
  v29 = (__int64)v20;
  v30 = 24;
  v31 = v26;
  return sub_3874770((__int64)a1, v30, v29, v31, v27, v28, *(double *)a3.m128_u64, a4, a5);
}
