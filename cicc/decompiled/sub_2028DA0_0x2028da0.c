// Function: sub_2028DA0
// Address: 0x2028da0
//
unsigned __int64 __fastcall sub_2028DA0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  char *v13; // rax
  char v14; // dl
  __int64 v15; // rax
  __m128i v16; // kr00_16
  __m128i v17; // kr10_16
  int v18; // eax
  char v19; // di
  unsigned int v20; // ebx
  unsigned __int64 result; // rax
  int v22; // eax
  __int32 v23; // eax
  __int64 v24; // rdx
  int v25; // eax
  unsigned int v26; // esi
  __int8 v27; // al
  __int64 v28; // rbx
  int v29; // eax
  __int32 v30; // edx
  int v31; // ecx
  char v32; // di
  char v33; // al
  unsigned __int32 v34; // edx
  unsigned int v35; // ecx
  const void **v36; // rsi
  unsigned int v37; // eax
  unsigned int v38; // ebx
  __int64 v39; // rdx
  char v40; // di
  int v41; // ecx
  unsigned __int8 v42; // al
  unsigned int v43; // ecx
  __int64 v44; // r15
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 *v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __m128i v51; // xmm0
  __m128i v52; // xmm1
  int v53; // edx
  int v54; // edx
  __int64 v55; // rdx
  unsigned __int8 v56; // al
  const void **v57; // rdx
  __int64 v58; // [rsp-10h] [rbp-150h]
  __int32 v59; // [rsp+8h] [rbp-138h]
  unsigned int v60; // [rsp+8h] [rbp-138h]
  int v61; // [rsp+8h] [rbp-138h]
  __int32 v62; // [rsp+8h] [rbp-138h]
  unsigned int v63; // [rsp+10h] [rbp-130h]
  __int64 v64; // [rsp+10h] [rbp-130h]
  unsigned int v65; // [rsp+10h] [rbp-130h]
  _QWORD *v66; // [rsp+18h] [rbp-128h]
  __int64 v69; // [rsp+70h] [rbp-D0h] BYREF
  int v70; // [rsp+78h] [rbp-C8h]
  char v71[8]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v72; // [rsp+88h] [rbp-B8h]
  _QWORD v73[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v74; // [rsp+A0h] [rbp-A0h] BYREF
  const void **v75; // [rsp+A8h] [rbp-98h]
  _QWORD v76[2]; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v77; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v78; // [rsp+D0h] [rbp-70h] BYREF
  _QWORD v79[2]; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v80; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v81; // [rsp+100h] [rbp-40h] BYREF

  v9 = *(_QWORD *)(a2 + 72);
  v69 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v69, v9, 2);
  v10 = a1[1];
  v70 = *(_DWORD *)(a2 + 64);
  v11 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v12 = *(_BYTE *)v11;
  v72 = *(_QWORD *)(v11 + 8);
  v13 = *(char **)(a2 + 40);
  v71[0] = v12;
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  LOBYTE(v73[0]) = v14;
  v73[1] = v15;
  sub_1D19A30((__int64)&v80, v10, v73);
  v16 = v80;
  v17 = v81;
  if ( v71[0] )
  {
    if ( (word_4305480[(unsigned __int8)(v71[0] - 14)] & 1) != 0 )
      goto LABEL_7;
    v18 = sub_2021900(v71[0]);
    v19 = v73[0];
    v20 = 2 * v18;
    if ( !LOBYTE(v73[0]) )
    {
LABEL_6:
      if ( v20 >= (unsigned int)sub_1F58D40((__int64)v73) )
        goto LABEL_7;
      goto LABEL_13;
    }
  }
  else
  {
    if ( (sub_1F58D30((__int64)v71) & 1) != 0 )
      goto LABEL_7;
    v22 = sub_1F58D40((__int64)v71);
    v19 = v73[0];
    v20 = 2 * v22;
    if ( !LOBYTE(v73[0]) )
      goto LABEL_6;
  }
  if ( v20 >= (unsigned int)sub_2021900(v19) )
    goto LABEL_7;
LABEL_13:
  v66 = *(_QWORD **)(a1[1] + 48);
  LOBYTE(v23) = sub_1F7E0F0((__int64)v71);
  v80.m128i_i32[0] = v23;
  v80.m128i_i64[1] = v24;
  if ( (_BYTE)v23 )
    v25 = sub_2021900(v23);
  else
    v25 = sub_1F58D40((__int64)&v80);
  v26 = 2 * v25;
  if ( 2 * v25 == 32 )
  {
    v27 = 5;
    goto LABEL_19;
  }
  if ( v26 > 0x20 )
  {
    if ( v26 == 64 )
    {
      v27 = 6;
      goto LABEL_19;
    }
    if ( v26 == 128 )
    {
      v27 = 7;
      goto LABEL_19;
    }
  }
  else
  {
    if ( v26 == 8 )
    {
      v27 = 3;
      goto LABEL_19;
    }
    v27 = 4;
    if ( v26 == 16 )
    {
LABEL_19:
      v28 = 0;
      goto LABEL_20;
    }
  }
  v27 = sub_1F58CC0(v66, v26);
  v28 = v55;
LABEL_20:
  v80.m128i_i8[0] = v27;
  v80.m128i_i64[1] = v28;
  if ( !v71[0] )
  {
    v29 = sub_1F58D30((__int64)v71);
    v30 = v80.m128i_i32[0];
    v31 = v29;
    v32 = v80.m128i_i8[0];
LABEL_22:
    v59 = v30;
    v63 = v31;
    v33 = sub_1D15020(v32, v31);
    v34 = v59;
    v35 = v63;
    goto LABEL_23;
  }
  v31 = word_4305480[(unsigned __int8)(v71[0] - 14)];
  v30 = v80.m128i_i32[0];
  v32 = v80.m128i_i8[0];
  if ( (unsigned __int8)(v71[0] - 56) > 0x1Du && (unsigned __int8)(v71[0] - 98) > 0xBu )
    goto LABEL_22;
  v62 = v80.m128i_i32[0];
  v65 = word_4305480[(unsigned __int8)(v71[0] - 14)];
  v33 = sub_1D154A0(v80.m128i_i8[0], v31);
  v35 = v65;
  v34 = v62;
LABEL_23:
  v36 = 0;
  if ( !v33 )
  {
    v33 = sub_1F593D0(v66, v34, v28, v35);
    v36 = v57;
  }
  LOBYTE(v74) = v33;
  v75 = v36;
  LOBYTE(v37) = sub_1F7E0F0((__int64)v71);
  v38 = v37;
  v64 = v39;
  if ( !v71[0] )
  {
    v40 = v37;
    v41 = (unsigned int)sub_1F58D30((__int64)v71) >> 1;
LABEL_27:
    v60 = v41;
    v42 = sub_1D15020(v40, v41);
    v43 = v60;
    v44 = v42;
    goto LABEL_28;
  }
  v40 = v37;
  v41 = word_4305480[(unsigned __int8)(v71[0] - 14)] >> 1;
  if ( (unsigned __int8)(v71[0] - 56) > 0x1Du && (unsigned __int8)(v71[0] - 98) > 0xBu )
    goto LABEL_27;
  v61 = word_4305480[(unsigned __int8)(v71[0] - 14)] >> 1;
  v56 = sub_1D154A0(v37, v41);
  v43 = v61;
  v44 = v56;
LABEL_28:
  if ( !(_BYTE)v44 )
    v44 = (unsigned __int8)sub_1F593D0(v66, v38, v64, v43);
  sub_1D19A30((__int64)&v80, a1[1], &v74);
  if ( v71[0] )
  {
    v45 = *a1;
    if ( *(_QWORD *)(*a1 + 8LL * (unsigned __int8)v71[0] + 120) )
    {
      if ( (!(_BYTE)v44 || !*(_QWORD *)(v45 + 8 * v44 + 120))
        && (_BYTE)v74
        && *(_QWORD *)(v45 + 8LL * (unsigned __int8)v74 + 120)
        && v80.m128i_i8[0]
        && *(_QWORD *)(v45 + 8LL * v80.m128i_u8[0] + 120) )
      {
        v46 = sub_1D309E0(
                (__int64 *)a1[1],
                *(unsigned __int16 *)(a2 + 24),
                (__int64)&v69,
                (unsigned int)v74,
                v75,
                0,
                *(double *)a5.m128i_i64,
                *(double *)a6.m128i_i64,
                *(double *)a7.m128i_i64,
                *(_OWORD *)*(_QWORD *)(a2 + 32));
        v47 = (__int64 *)a1[1];
        v77.m128i_i8[0] = 0;
        v76[0] = v46;
        v76[1] = v48;
        v77.m128i_i64[1] = 0;
        v49 = *(_QWORD *)(v46 + 40) + 16LL * (unsigned int)v48;
        v78.m128i_i64[1] = 0;
        v78.m128i_i8[0] = 0;
        LOBYTE(v48) = *(_BYTE *)v49;
        v50 = *(_QWORD *)(v49 + 8);
        LOBYTE(v79[0]) = v48;
        v79[1] = v50;
        sub_1D19A30((__int64)&v80, (__int64)v47, v79);
        v51 = _mm_loadu_si128(&v80);
        v52 = _mm_loadu_si128(&v81);
        v77 = v51;
        v78 = v52;
        sub_1D40600(
          (__int64)&v80,
          v47,
          (__int64)v76,
          (__int64)&v69,
          (const void ***)&v77,
          (const void ***)&v78,
          v51,
          *(double *)v52.m128i_i64,
          a7);
        *(_QWORD *)a3 = v80.m128i_i64[0];
        *(_DWORD *)(a3 + 8) = v80.m128i_i32[2];
        *(_QWORD *)a4 = v81.m128i_i64[0];
        *(_DWORD *)(a4 + 8) = v81.m128i_i32[2];
        *(_QWORD *)a3 = sub_1D309E0(
                          (__int64 *)a1[1],
                          *(unsigned __int16 *)(a2 + 24),
                          (__int64)&v69,
                          v16.m128i_i64[0],
                          (const void **)v16.m128i_i64[1],
                          0,
                          *(double *)v51.m128i_i64,
                          *(double *)v52.m128i_i64,
                          *(double *)a7.m128i_i64,
                          *(_OWORD *)a3);
        *(_DWORD *)(a3 + 8) = v53;
        *(_QWORD *)a4 = sub_1D309E0(
                          (__int64 *)a1[1],
                          *(unsigned __int16 *)(a2 + 24),
                          (__int64)&v69,
                          v17.m128i_i64[0],
                          (const void **)v17.m128i_i64[1],
                          0,
                          *(double *)v51.m128i_i64,
                          *(double *)v52.m128i_i64,
                          *(double *)a7.m128i_i64,
                          *(_OWORD *)a4);
        *(_DWORD *)(a4 + 8) = v54;
        sub_17CD270(&v69);
        return v58;
      }
    }
  }
LABEL_7:
  result = sub_2028A10((__int64)a1, a2, a3, a4, a5, a6, a7);
  if ( v69 )
    return sub_161E7C0((__int64)&v69, v69);
  return result;
}
