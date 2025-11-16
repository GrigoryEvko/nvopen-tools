// Function: sub_21334C0
// Address: 0x21334c0
//
unsigned __int64 __fastcall sub_21334C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v11; // rsi
  __int64 *v12; // rdi
  int v13; // edx
  __int64 *v14; // r15
  __int64 v15; // rax
  unsigned int v16; // edx
  unsigned __int8 v17; // al
  char v18; // di
  __int64 v19; // rcx
  unsigned int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned int *v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rcx
  const void **v27; // r8
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // edx
  unsigned __int64 result; // rax
  unsigned __int8 v33; // al
  __int128 v34; // [rsp-20h] [rbp-C0h]
  unsigned __int64 v35; // [rsp-10h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-98h]
  unsigned int v37; // [rsp+40h] [rbp-60h] BYREF
  const void **v38; // [rsp+48h] [rbp-58h]
  __int64 v39; // [rsp+50h] [rbp-50h] BYREF
  int v40; // [rsp+58h] [rbp-48h]
  const void **v41; // [rsp+60h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v39,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 72);
  LOBYTE(v37) = v40;
  v39 = v11;
  v38 = v41;
  if ( v11 )
    sub_1623A60((__int64)&v39, v11, 2);
  v12 = *(__int64 **)(a1 + 8);
  v40 = *(_DWORD *)(a2 + 64);
  *(_QWORD *)a3 = sub_1D309E0(
                    v12,
                    145,
                    (__int64)&v39,
                    v37,
                    v38,
                    0,
                    *(double *)a5.m128i_i64,
                    a6,
                    *(double *)a7.m128i_i64,
                    *(_OWORD *)*(_QWORD *)(a2 + 32));
  *(_DWORD *)(a3 + 8) = v13;
  v14 = *(__int64 **)(a1 + 8);
  v15 = sub_1E0A0C0(v14[4]);
  v16 = 8 * sub_15A9520(v15, 0);
  if ( v16 == 32 )
  {
    v17 = 5;
  }
  else if ( v16 > 0x20 )
  {
    v17 = 6;
    if ( v16 != 64 )
    {
      v33 = 0;
      v18 = v37;
      if ( v16 == 128 )
        v33 = 7;
      v19 = v33;
      if ( !(_BYTE)v37 )
        goto LABEL_8;
      goto LABEL_16;
    }
  }
  else
  {
    v17 = 3;
    if ( v16 != 8 )
      v17 = 4 * (v16 == 16);
  }
  v18 = v37;
  v19 = v17;
  if ( !(_BYTE)v37 )
  {
LABEL_8:
    v36 = v19;
    v20 = sub_1F58D40((__int64)&v37);
    v21 = v36;
    goto LABEL_9;
  }
LABEL_16:
  v20 = sub_2127930(v18);
LABEL_9:
  v22 = sub_1D38BB0((__int64)v14, v20, (__int64)&v39, v21, 0, 0, a5, a6, a7, 0);
  v23 = *(unsigned int **)(a2 + 32);
  *((_QWORD *)&v34 + 1) = v24;
  *(_QWORD *)&v34 = v22;
  v25 = sub_1D332F0(
          v14,
          124,
          (__int64)&v39,
          *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v23 + 40LL) + 16LL * v23[2]),
          *(const void ***)(*(_QWORD *)(*(_QWORD *)v23 + 40LL) + 16LL * v23[2] + 8),
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          *(_QWORD *)v23,
          *((_QWORD *)v23 + 1),
          v34);
  v26 = v37;
  v27 = v38;
  *(_QWORD *)a4 = v25;
  *(_DWORD *)(a4 + 8) = v28;
  v29 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          145,
          (__int64)&v39,
          v26,
          v27,
          0,
          *(double *)a5.m128i_i64,
          a6,
          *(double *)a7.m128i_i64,
          *(_OWORD *)a4);
  v30 = v39;
  *(_QWORD *)a4 = v29;
  *(_DWORD *)(a4 + 8) = v31;
  result = v35;
  if ( v30 )
    return sub_161E7C0((__int64)&v39, v30);
  return result;
}
