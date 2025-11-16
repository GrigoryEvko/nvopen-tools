// Function: sub_213D870
// Address: 0x213d870
//
unsigned __int64 __fastcall sub_213D870(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v9; // rsi
  unsigned __int8 v10; // r14
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  char v15; // di
  const void **v16; // rax
  int v17; // edx
  _QWORD *v18; // rdi
  __int64 v19; // r9
  _QWORD *v20; // r12
  unsigned int v21; // edx
  unsigned int v22; // r13d
  unsigned __int64 result; // rax
  unsigned int v24; // eax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int128 v27; // [rsp-10h] [rbp-C0h]
  unsigned int v28; // [rsp+4h] [rbp-ACh]
  const void **v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+40h] [rbp-70h] BYREF
  int v33; // [rsp+48h] [rbp-68h]
  char v34[8]; // [rsp+50h] [rbp-60h] BYREF
  const void **v35; // [rsp+58h] [rbp-58h]
  __int64 v36; // [rsp+60h] [rbp-50h] BYREF
  const void **v37; // [rsp+68h] [rbp-48h]
  const void **v38; // [rsp+70h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v36,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v9 = *(_QWORD *)(a2 + 72);
  v10 = (unsigned __int8)v37;
  v31 = v38;
  v32 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v32, v9, 2);
  v33 = *(_DWORD *)(a2 + 64);
  v11 = *(unsigned __int64 **)(a2 + 32);
  v12 = *v11;
  v13 = v11[1];
  v14 = *(_QWORD *)(*v11 + 40) + 16LL * *((unsigned int *)v11 + 2);
  v15 = *(_BYTE *)v14;
  v16 = *(const void ***)(v14 + 8);
  LOBYTE(v36) = v10;
  v37 = v31;
  v34[0] = v15;
  v35 = v16;
  if ( v15 == v10 )
  {
    if ( v10 || v16 == v31 )
      goto LABEL_5;
LABEL_17:
    v28 = sub_1F58D40((__int64)v34);
    if ( !v10 )
      goto LABEL_18;
LABEL_13:
    v24 = sub_2127930(v10);
    goto LABEL_14;
  }
  if ( !v15 )
    goto LABEL_17;
  v28 = sub_2127930(v15);
  if ( v10 )
    goto LABEL_13;
LABEL_18:
  v24 = sub_1F58D40((__int64)&v36);
LABEL_14:
  if ( v24 < v28 )
  {
    v25 = sub_2138AD0(a1, v12, v13);
    result = sub_200E870(a1, v25, v26, a3, (_QWORD *)a4, a5, a6, a7);
    goto LABEL_8;
  }
LABEL_5:
  *((_QWORD *)&v27 + 1) = v13;
  *(_QWORD *)&v27 = v12;
  *(_QWORD *)a3 = sub_1D309E0(
                    *(__int64 **)(a1 + 8),
                    144,
                    (__int64)&v32,
                    v10,
                    v31,
                    0,
                    *(double *)a5.m128i_i64,
                    a6,
                    *(double *)a7.m128i_i64,
                    v27);
  v36 = 0;
  *(_DWORD *)(a3 + 8) = v17;
  v18 = *(_QWORD **)(a1 + 8);
  LODWORD(v37) = 0;
  v20 = sub_1D2B300(v18, 0x30u, (__int64)&v36, v10, (__int64)v31, v19);
  v22 = v21;
  if ( v36 )
    sub_161E7C0((__int64)&v36, v36);
  *(_QWORD *)a4 = v20;
  result = v22;
  *(_DWORD *)(a4 + 8) = v22;
LABEL_8:
  if ( v32 )
    return sub_161E7C0((__int64)&v32, v32);
  return result;
}
