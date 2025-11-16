// Function: sub_3409320
// Address: 0x3409320
//
unsigned __int8 *__fastcall sub_3409320(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        __m128i a7,
        int a8)
{
  __int64 v13; // r9
  unsigned __int16 *v14; // r10
  unsigned __int16 v15; // dx
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int8 *v19; // rax
  unsigned int v20; // edx
  unsigned __int8 *v21; // rcx
  __int64 v22; // r8
  unsigned int v24; // edx
  unsigned __int16 *v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  __int64 v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v29; // [rsp+18h] [rbp-88h]
  unsigned __int64 v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+58h] [rbp-48h]
  __int64 v32; // [rsp+60h] [rbp-40h]
  __int64 v33; // [rsp+68h] [rbp-38h]

  v13 = 16LL * (unsigned int)a3;
  v14 = (unsigned __int16 *)(v13 + *(_QWORD *)(a2 + 48));
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  if ( !a5 )
  {
    v21 = sub_3400BD0((__int64)a1, a4, a6, *v14, v16, 0, a7, 0);
    v22 = v24;
    return sub_34092D0(a1, a2, a3, (__int64)v21, v22, a6, a7, a8);
  }
  LOWORD(v30) = *v14;
  v31 = v16;
  if ( v15 )
  {
    if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
      BUG();
    LODWORD(v31) = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
    if ( (unsigned int)v31 <= 0x40 )
      goto LABEL_4;
LABEL_13:
    v28 = v13;
    sub_C43690((__int64)&v30, a4, 0);
    v14 = (unsigned __int16 *)(v28 + *(_QWORD *)(a2 + 48));
    goto LABEL_5;
  }
  v25 = v14;
  v26 = v13;
  v17 = sub_3007260((__int64)&v30);
  v14 = v25;
  v32 = v17;
  v13 = v26;
  v33 = v18;
  LODWORD(v31) = v17;
  if ( (unsigned int)v17 > 0x40 )
    goto LABEL_13;
LABEL_4:
  v30 = a4;
LABEL_5:
  v19 = sub_3401900((__int64)a1, a6, *v14, *((_QWORD *)v14 + 1), (__int64)&v30, 1, a7);
  v21 = v19;
  v22 = v20;
  if ( (unsigned int)v31 > 0x40 && v30 )
  {
    v27 = v20;
    v29 = v19;
    j_j___libc_free_0_0(v30);
    v21 = v29;
    v22 = v27;
  }
  return sub_34092D0(a1, a2, a3, (__int64)v21, v22, a6, a7, a8);
}
