// Function: sub_202D8A0
// Address: 0x202d8a0
//
__int64 *__fastcall sub_202D8A0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rsi
  char *v11; // rax
  char v12; // bl
  char *v13; // rax
  char v14; // r15
  const void **v15; // rax
  char v16; // di
  unsigned int v17; // ebx
  char v18; // di
  __int64 v19; // rsi
  int v20; // eax
  unsigned int v21; // ebx
  unsigned __int8 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r12
  char v25; // al
  __int64 v26; // rdx
  const void **v27; // r15
  unsigned int v28; // ebx
  unsigned int v29; // eax
  unsigned __int8 v30; // r10
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 v35; // r15
  _QWORD *v36; // rbx
  __int64 v37; // rax
  const void **v38; // r8
  __int64 v39; // rcx
  __int128 v40; // rax
  __int128 v41; // kr00_16
  __int64 *v42; // r12
  __int64 v43; // rax
  unsigned int v44; // edx
  unsigned __int8 v45; // al
  __int128 v46; // rax
  __int64 v48; // rdx
  const void **v49; // rdx
  const void **v50; // rdx
  __int128 v51; // [rsp-10h] [rbp-D0h]
  __int64 v52; // [rsp+0h] [rbp-C0h]
  _QWORD *v53; // [rsp+8h] [rbp-B8h]
  __int64 v54; // [rsp+10h] [rbp-B0h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v56; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v57; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v58; // [rsp+20h] [rbp-A0h]
  __int64 v59; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v60; // [rsp+28h] [rbp-98h]
  bool v61; // [rsp+33h] [rbp-8Dh]
  unsigned int v62; // [rsp+34h] [rbp-8Ch]
  __int64 v63; // [rsp+40h] [rbp-80h] BYREF
  __int64 v64; // [rsp+48h] [rbp-78h]
  __int64 v65; // [rsp+50h] [rbp-70h] BYREF
  const void **v66; // [rsp+58h] [rbp-68h]
  __int64 v67; // [rsp+60h] [rbp-60h] BYREF
  int v68; // [rsp+68h] [rbp-58h]
  __int128 v69; // [rsp+70h] [rbp-50h] BYREF
  __int128 v70[4]; // [rsp+80h] [rbp-40h] BYREF

  v8 = *(unsigned __int64 **)(a2 + 32);
  v9 = *v8;
  v10 = v8[1];
  v11 = *(char **)(*v8 + 40);
  v58 = v9;
  v55 = v10;
  v12 = *v11;
  v64 = *((_QWORD *)v11 + 1);
  v13 = *(char **)(a2 + 40);
  LOBYTE(v63) = v12;
  v14 = *v13;
  v15 = (const void **)*((_QWORD *)v13 + 1);
  LOBYTE(v65) = v14;
  v66 = v15;
  if ( v14 )
  {
    v62 = word_4305480[(unsigned __int8)(v14 - 14)];
    v61 = (unsigned __int8)(v14 - 86) <= 0x17u || (unsigned __int8)(v14 - 8) <= 5u;
  }
  else
  {
    v62 = sub_1F58D30((__int64)&v65);
    v61 = sub_1F58CD0((__int64)&v65);
  }
  if ( v12 )
  {
    if ( (unsigned __int8)(v12 - 14) > 0x5Fu )
      goto LABEL_5;
LABEL_23:
    v25 = sub_1F7E0F0((__int64)&v63);
    v14 = v65;
    v16 = v25;
    LOBYTE(v70[0]) = v25;
    *((_QWORD *)&v70[0] + 1) = v26;
    if ( v25 )
      goto LABEL_6;
    goto LABEL_24;
  }
  if ( sub_1F58D20((__int64)&v63) )
    goto LABEL_23;
LABEL_5:
  v16 = v63;
  LOBYTE(v70[0]) = v63;
  *((_QWORD *)&v70[0] + 1) = v64;
  if ( (_BYTE)v63 )
  {
LABEL_6:
    v17 = sub_2021900(v16);
    goto LABEL_7;
  }
LABEL_24:
  v17 = sub_1F58D40((__int64)v70);
LABEL_7:
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_9;
  }
  else if ( !sub_1F58D20((__int64)&v65) )
  {
LABEL_9:
    v18 = v65;
    LOBYTE(v70[0]) = v65;
    *((_QWORD *)&v70[0] + 1) = v66;
    if ( (_BYTE)v65 )
      goto LABEL_10;
    goto LABEL_20;
  }
  v18 = sub_1F7E0F0((__int64)&v65);
  *((_QWORD *)&v70[0] + 1) = v23;
  LOBYTE(v70[0]) = v18;
  if ( v18 )
  {
LABEL_10:
    if ( 2 * (unsigned int)sub_2021900(v18) < v17 )
      goto LABEL_11;
    return sub_202A670(a1, a2, *(double *)a3.m128i_i64, a4, a5);
  }
LABEL_20:
  if ( 2 * (unsigned int)sub_1F58D40((__int64)v70) >= v17 )
    return sub_202A670(a1, a2, *(double *)a3.m128i_i64, a4, a5);
LABEL_11:
  v19 = *(_QWORD *)(a2 + 72);
  v67 = v19;
  if ( v19 )
    sub_1623A60((__int64)&v67, v19, 2);
  v20 = *(_DWORD *)(a2 + 64);
  v21 = v17 >> 1;
  *(_QWORD *)&v69 = 0;
  v68 = v20;
  DWORD2(v69) = 0;
  *(_QWORD *)&v70[0] = 0;
  DWORD2(v70[0]) = 0;
  sub_2017DE0(a1, v58, v55, &v69, v70);
  if ( v61 )
  {
    if ( v21 == 64 )
    {
      v22 = 10;
    }
    else if ( v21 <= 0x40 )
    {
      v22 = (v21 != 16) + 8;
    }
    else
    {
      v22 = (v21 != 80) + 11;
    }
    v54 = 0;
    v53 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
    goto LABEL_30;
  }
  v53 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  if ( v21 == 32 )
  {
    v22 = 5;
  }
  else if ( v21 > 0x20 )
  {
    if ( v21 == 64 )
    {
      v22 = 6;
    }
    else
    {
      if ( v21 != 128 )
      {
LABEL_54:
        v22 = sub_1F58CC0(v53, v21);
        v54 = v48;
        v53 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
        goto LABEL_30;
      }
      v22 = 7;
    }
  }
  else if ( v21 == 8 )
  {
    v22 = 3;
  }
  else
  {
    v22 = 4;
    if ( v21 != 16 )
    {
      v22 = 2;
      if ( v21 != 1 )
        goto LABEL_54;
    }
  }
  v54 = 0;
LABEL_30:
  v56 = v22;
  v27 = 0;
  v28 = v22;
  LOBYTE(v29) = sub_1D15020(v22, v62 >> 1);
  v30 = v56;
  if ( !(_BYTE)v29 )
  {
    v29 = sub_1F593D0(v53, v28, v54, v62 >> 1);
    v30 = v56;
    v5 = v29;
    v27 = v50;
  }
  LOBYTE(v5) = v29;
  v57 = v30;
  v31 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v67,
          v5,
          v27,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          v69);
  v60 = v32;
  v59 = v31;
  v33 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v67,
          v5,
          v27,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          v70[0]);
  v35 = v34;
  v36 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v37) = sub_1D15020(v57, v62);
  v38 = 0;
  if ( !(_BYTE)v37 )
  {
    v37 = sub_1F593D0(v36, v57, v54, v62);
    v52 = v37;
    v38 = v49;
  }
  v39 = v52;
  *((_QWORD *)&v51 + 1) = v35;
  *(_QWORD *)&v51 = v33;
  LOBYTE(v39) = v37;
  *(_QWORD *)&v40 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      107,
                      (__int64)&v67,
                      v39,
                      v38,
                      0,
                      *(double *)a3.m128i_i64,
                      a4,
                      a5,
                      v59,
                      v60,
                      v51);
  v41 = v40;
  if ( v61 )
  {
    v42 = *(__int64 **)(a1 + 8);
    v43 = sub_1E0A0C0(v42[4]);
    v44 = 8 * sub_15A9520(v43, 0);
    if ( v44 == 32 )
    {
      v45 = 5;
    }
    else if ( v44 > 0x20 )
    {
      v45 = 6;
      if ( v44 != 64 )
      {
        v45 = 0;
        if ( v44 == 128 )
          v45 = 7;
      }
    }
    else
    {
      v45 = 3;
      if ( v44 != 8 )
        v45 = 4 * (v44 == 16);
    }
    *(_QWORD *)&v46 = sub_1D38BB0((__int64)v42, 0, (__int64)&v67, v45, 0, 1, a3, a4, a5, 0);
    v24 = (__int64)sub_1D332F0(
                     v42,
                     154,
                     (__int64)&v67,
                     (unsigned int)v65,
                     v66,
                     0,
                     *(double *)a3.m128i_i64,
                     a4,
                     a5,
                     v41,
                     *((unsigned __int64 *)&v41 + 1),
                     v46);
  }
  else
  {
    v24 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            145,
            (__int64)&v67,
            (unsigned int)v65,
            v66,
            0,
            *(double *)a3.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            v40);
  }
  if ( v67 )
    sub_161E7C0((__int64)&v67, v67);
  return (__int64 *)v24;
}
