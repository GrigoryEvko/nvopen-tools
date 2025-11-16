// Function: sub_1F732B0
// Address: 0x1f732b0
//
__int64 *__fastcall sub_1F732B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  unsigned __int64 v9; // r11
  unsigned __int8 *v13; // rax
  __int64 v14; // r13
  unsigned int v15; // r14d
  _DWORD *v16; // rdi
  __int64 v17; // rax
  __int64 *result; // rax
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rsi
  __int64 *v23; // r15
  __int128 v24; // rax
  __int64 v25; // rsi
  const void ***v26; // rcx
  __int128 v27; // kr00_16
  int v28; // r8d
  bool v29; // al
  const __m128i *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // r10
  unsigned __int8 *v33; // rax
  const void **v34; // r8
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // r11
  __int64 v39; // rcx
  __int64 v40; // r8
  int v41; // eax
  __int64 v42; // r9
  __int64 v43; // rsi
  __int64 *v44; // r13
  __int64 v45; // r14
  const void ***v46; // r15
  int v47; // r12d
  __int128 v48; // [rsp-10h] [rbp-B0h]
  __int64 v49; // [rsp+0h] [rbp-A0h]
  __int64 v50; // [rsp+8h] [rbp-98h]
  const void ***v51; // [rsp+10h] [rbp-90h]
  const void **v52; // [rsp+10h] [rbp-90h]
  __int64 v53; // [rsp+10h] [rbp-90h]
  int v54; // [rsp+18h] [rbp-88h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  unsigned __int64 v56; // [rsp+18h] [rbp-88h]
  unsigned __int64 v57; // [rsp+18h] [rbp-88h]
  __int128 v58; // [rsp+20h] [rbp-80h]
  const void **v60; // [rsp+38h] [rbp-68h]
  __int64 *v61; // [rsp+38h] [rbp-68h]
  __int128 v62; // [rsp+40h] [rbp-60h]
  __int64 *v63; // [rsp+40h] [rbp-60h]
  __int64 *v64; // [rsp+40h] [rbp-60h]
  __int64 v65; // [rsp+50h] [rbp-50h] BYREF
  int v66; // [rsp+58h] [rbp-48h]
  __int64 v67; // [rsp+60h] [rbp-40h] BYREF
  int v68; // [rsp+68h] [rbp-38h]

  v9 = a5;
  *((_QWORD *)&v62 + 1) = a3;
  *(_QWORD *)&v62 = a2;
  v13 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v14 = *v13;
  v15 = (unsigned __int8)v14;
  v60 = (const void **)*((_QWORD *)v13 + 1);
  if ( *(_WORD *)(a4 + 24) != 68 )
    goto LABEL_2;
  v29 = sub_1D185B0(*(_QWORD *)(*(_QWORD *)(a4 + 32) + 40LL));
  v9 = a5;
  if ( !v29 )
    goto LABEL_2;
  v30 = *(const __m128i **)(a4 + 32);
  v31 = *(_QWORD *)(a6 + 72);
  v32 = *a1;
  a7 = _mm_loadu_si128(v30);
  v33 = (unsigned __int8 *)(*(_QWORD *)(v30->m128i_i64[0] + 40) + 16LL * v30->m128i_u32[2]);
  v34 = (const void **)*((_QWORD *)v33 + 1);
  v35 = *v33;
  v67 = v31;
  if ( v31 )
  {
    v49 = v35;
    v52 = v34;
    v55 = v32;
    sub_1623A60((__int64)&v67, v31, 2);
    v35 = v49;
    v9 = a5;
    v34 = v52;
    v32 = v55;
  }
  v56 = v9;
  v68 = *(_DWORD *)(a6 + 64);
  v36 = sub_1D38BB0(v32, 1, (__int64)&v67, v35, v34, 0, a7, a8, a9, 0);
  v38 = v56;
  v39 = v36;
  v40 = v37;
  if ( v67 )
  {
    v50 = v37;
    v53 = v36;
    sub_161E7C0((__int64)&v67, v67);
    v40 = v50;
    v39 = v53;
    v38 = v56;
  }
  v57 = v38;
  v41 = sub_1D1FED0(*a1, a7.m128i_i64[0], a7.m128i_i64[1], v39, v40);
  v9 = v57;
  if ( !v41 )
  {
    v43 = *(_QWORD *)(a6 + 72);
    v44 = (__int64 *)*a1;
    v45 = *(_QWORD *)(a4 + 32);
    v46 = *(const void ****)(a6 + 40);
    v67 = v43;
    v47 = *(_DWORD *)(a6 + 60);
    if ( v43 )
      sub_1623A60((__int64)&v67, v43, 2);
    v68 = *(_DWORD *)(a6 + 64);
    result = sub_1D37470(v44, 68, (__int64)&v67, v46, v47, v42, v62, *(_OWORD *)&a7, *(_OWORD *)(v45 + 80));
    if ( v67 )
    {
      v64 = result;
      sub_161E7C0((__int64)&v67, v67);
      return v64;
    }
  }
  else
  {
LABEL_2:
    v16 = (_DWORD *)a1[1];
    v17 = 1;
    if ( ((_BYTE)v14 == 1 || (_BYTE)v14 && (v17 = (unsigned __int8)v14, *(_QWORD *)&v16[2 * v14 + 30]))
      && (*((_BYTE *)v16 + 259 * v17 + 2490) & 0xFB) == 0
      && (v19 = sub_1F6DE40(v16, a4, v9), v21 = v20, v19) )
    {
      v22 = *(_QWORD *)(a6 + 72);
      v23 = (__int64 *)*a1;
      v67 = v22;
      if ( v22 )
        sub_1623A60((__int64)&v67, v22, 2);
      v68 = *(_DWORD *)(a6 + 64);
      *(_QWORD *)&v24 = sub_1D38BB0((__int64)v23, 0, (__int64)&v67, v15, v60, 0, a7, a8, a9, 0);
      v25 = *(_QWORD *)(a6 + 72);
      v26 = *(const void ****)(a6 + 40);
      v27 = v24;
      v28 = *(_DWORD *)(a6 + 60);
      v65 = v25;
      if ( v25 )
      {
        v58 = v24;
        v51 = v26;
        v54 = v28;
        sub_1623A60((__int64)&v65, v25, 2);
        v26 = v51;
        v28 = v54;
        v27 = v58;
      }
      *((_QWORD *)&v48 + 1) = v21;
      *(_QWORD *)&v48 = v19;
      v66 = *(_DWORD *)(a6 + 64);
      result = sub_1D37470(v23, 68, (__int64)&v65, v26, v28, (__int64)&v65, v62, v27, v48);
      if ( v65 )
      {
        v61 = result;
        sub_161E7C0((__int64)&v65, v65);
        result = v61;
      }
      if ( v67 )
      {
        v63 = result;
        sub_161E7C0((__int64)&v67, v67);
        return v63;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
