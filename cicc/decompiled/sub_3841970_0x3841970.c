// Function: sub_3841970
// Address: 0x3841970
//
unsigned __int8 *__fastcall sub_3841970(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v5; // rsi
  __int64 v6; // r8
  unsigned int v7; // edx
  __int64 v8; // r13
  unsigned int *v9; // rax
  __int64 v10; // rcx
  unsigned __int16 *v11; // rdx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int16 *v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  unsigned int v18; // r8d
  __int16 *v19; // rax
  unsigned int v20; // r14d
  __int16 v21; // r11
  __int64 v22; // r9
  __int64 v23; // rsi
  __int64 v24; // r12
  unsigned __int8 *v25; // r12
  char v27; // al
  unsigned __int16 v28; // r10
  char v29; // al
  __int16 v30; // r10
  unsigned __int16 v31; // dx
  unsigned int v32; // eax
  __m128i v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // r9d
  char v37; // al
  __int64 v38; // r10
  char v39; // al
  unsigned int v40; // eax
  __int64 v41; // rdx
  char v42; // al
  unsigned __int16 v43; // r10
  char v44; // al
  __int16 v45; // r10
  unsigned __int16 v46; // dx
  unsigned int v47; // eax
  char v48; // r14
  bool v49; // al
  char v50; // r14
  bool v51; // al
  __int64 v52; // [rsp+0h] [rbp-D0h]
  __int64 v53; // [rsp+0h] [rbp-D0h]
  __int16 v54; // [rsp+Ch] [rbp-C4h]
  __int16 v55; // [rsp+Ch] [rbp-C4h]
  __int16 v56; // [rsp+Ch] [rbp-C4h]
  __int16 v57; // [rsp+Ch] [rbp-C4h]
  __int16 v58; // [rsp+Ch] [rbp-C4h]
  _DWORD *v59; // [rsp+10h] [rbp-C0h]
  __int64 v60; // [rsp+10h] [rbp-C0h]
  unsigned __int16 v61; // [rsp+10h] [rbp-C0h]
  _DWORD *v62; // [rsp+10h] [rbp-C0h]
  __int64 v63; // [rsp+10h] [rbp-C0h]
  __int64 v64; // [rsp+10h] [rbp-C0h]
  unsigned int v65; // [rsp+18h] [rbp-B8h]
  __int16 v66; // [rsp+18h] [rbp-B8h]
  unsigned int v67; // [rsp+18h] [rbp-B8h]
  __int16 v68; // [rsp+18h] [rbp-B8h]
  unsigned int v69; // [rsp+18h] [rbp-B8h]
  __int16 v70; // [rsp+18h] [rbp-B8h]
  __int16 v71; // [rsp+18h] [rbp-B8h]
  __int64 v72; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v73; // [rsp+30h] [rbp-A0h]
  __int64 v74; // [rsp+40h] [rbp-90h] BYREF
  int v75; // [rsp+48h] [rbp-88h]
  __m128i v76; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int16 v77[4]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v78; // [rsp+68h] [rbp-68h]
  unsigned __int16 v79; // [rsp+70h] [rbp-60h] BYREF
  __int64 v80; // [rsp+78h] [rbp-58h]
  unsigned __int64 v81; // [rsp+80h] [rbp-50h]
  __int64 v82; // [rsp+88h] [rbp-48h]
  __m128i v83; // [rsp+90h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v74 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v74, v5, 1);
  v75 = *(_DWORD *)(a2 + 72);
  v73 = sub_3841260((__int64)a1, a2, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
  v8 = v7;
  v9 = *(unsigned int **)(a2 + 40);
  v10 = *(_QWORD *)v9;
  v11 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v9 + 48LL) + 16LL * v9[2]);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v83.m128i_i16[0] = v12;
  v83.m128i_i64[1] = v13;
  if ( (_WORD)v12 )
    v14 = (unsigned __int16)word_4456580[v12 - 1];
  else
    v14 = (unsigned int)sub_3009970((__int64)&v83, a2, v13, v10, v6);
  v15 = (unsigned __int16 *)(*((_QWORD *)v73 + 6) + 16 * v8);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  v76.m128i_i16[0] = v16;
  v76.m128i_i64[1] = v17;
  if ( (_WORD)v16 )
  {
    v72 = 0;
    v18 = (unsigned __int16)word_4456580[v16 - 1];
  }
  else
  {
    v68 = v14;
    v40 = sub_3009970((__int64)&v76, a2, v17, v14, v6);
    LOWORD(v14) = v68;
    v72 = v41;
    HIWORD(v3) = HIWORD(v40);
    v18 = v40;
  }
  v19 = *(__int16 **)(a2 + 48);
  v20 = *(_DWORD *)(a2 + 24);
  LOWORD(v3) = v18;
  v21 = *v19;
  v22 = *((_QWORD *)v19 + 1);
  v77[0] = *v19;
  v78 = v22;
  if ( v20 != 386 )
  {
    if ( v20 != 385 )
    {
      if ( v20 != 384 )
        goto LABEL_11;
      if ( (_WORD)v14 != 2 )
        goto LABEL_11;
      v69 = v18;
      v62 = (_DWORD *)*a1;
      v42 = sub_3813820(*a1, 0x180u, v76.m128i_u16[0], 0, v18);
      LOWORD(v18) = v69;
      if ( v42 )
        goto LABEL_11;
      v44 = sub_3813820((__int64)v62, 0x186u, v43, 0, v69);
      LOWORD(v18) = v69;
      if ( !v44 )
        goto LABEL_11;
      v83 = _mm_loadu_si128(&v76);
      if ( v45 )
      {
        v46 = v45 - 17;
        if ( (unsigned __int16)(v45 - 10) > 6u && (unsigned __int16)(v45 - 126) > 0x31u )
        {
          if ( v46 > 0xD3u )
          {
LABEL_49:
            v47 = v62[15];
            goto LABEL_50;
          }
          goto LABEL_65;
        }
        if ( v46 <= 0xD3u )
        {
LABEL_65:
          v47 = v62[17];
LABEL_50:
          if ( v47 > 1 )
          {
            v20 = 390;
            if ( v47 == 2 )
            {
              v56 = v18;
              v20 = 390;
              v64 = v22;
              v71 = v21;
              sub_383B380((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
              LOWORD(v18) = v56;
              v22 = v64;
              v21 = v71;
            }
          }
          else
          {
            v55 = v18;
            v20 = 390;
            v63 = v22;
            v70 = v21;
            sub_37AF270((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
            v21 = v70;
            v22 = v63;
            LOWORD(v18) = v55;
          }
          goto LABEL_11;
        }
      }
      else
      {
        v53 = v22;
        v58 = v21;
        v50 = sub_3007030((__int64)&v83);
        v51 = sub_30070B0((__int64)&v83);
        v21 = v58;
        v22 = v53;
        LOWORD(v18) = v69;
        if ( v51 )
          goto LABEL_65;
        if ( !v50 )
          goto LABEL_49;
      }
      v47 = v62[16];
      goto LABEL_50;
    }
    if ( (_WORD)v14 != 2 )
      goto LABEL_11;
    v65 = v18;
    v59 = (_DWORD *)*a1;
    v27 = sub_3813820(*a1, 0x181u, v76.m128i_u16[0], 0, v18);
    LOWORD(v18) = v65;
    if ( v27 )
      goto LABEL_11;
    v29 = sub_3813820((__int64)v59, 0x185u, v28, 0, v65);
    LOWORD(v18) = v65;
    if ( !v29 )
      goto LABEL_11;
    a3 = _mm_loadu_si128(&v76);
    v83 = a3;
    if ( v30 )
    {
      v31 = v30 - 17;
      if ( (unsigned __int16)(v30 - 10) > 6u && (unsigned __int16)(v30 - 126) > 0x31u )
      {
        if ( v31 > 0xD3u )
        {
LABEL_27:
          v32 = v59[15];
          goto LABEL_28;
        }
        goto LABEL_63;
      }
      if ( v31 <= 0xD3u )
      {
LABEL_63:
        v32 = v59[17];
LABEL_28:
        if ( v32 > 1 )
        {
          v20 = 389;
          if ( v32 != 2 )
            goto LABEL_11;
          v54 = v18;
          v60 = v22;
          v66 = v21;
          sub_383B380((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
        }
        else
        {
          v54 = v18;
          v60 = v22;
          v66 = v21;
          sub_37AF270((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
        }
        v20 = 389;
        v21 = v66;
        v22 = v60;
        LOWORD(v18) = v54;
        goto LABEL_11;
      }
    }
    else
    {
      v52 = v22;
      v57 = v21;
      v48 = sub_3007030((__int64)&v83);
      v49 = sub_30070B0((__int64)&v83);
      v21 = v57;
      v22 = v52;
      LOWORD(v18) = v65;
      if ( v49 )
        goto LABEL_63;
      if ( !v48 )
        goto LABEL_27;
    }
    v32 = v59[16];
    goto LABEL_28;
  }
  if ( (_WORD)v14 == 2 )
  {
    v67 = v18;
    v61 = v76.m128i_i16[0];
    v37 = sub_3813820(*a1, 0x182u, v76.m128i_u16[0], 0, v18);
    LOWORD(v18) = v67;
    if ( !v37 )
    {
      v39 = sub_3813820(v38, 0x17Eu, v61, 0, v67);
      LOWORD(v18) = v67;
      if ( v39 )
        v20 = 382;
    }
  }
LABEL_11:
  if ( v21 == (_WORD)v18 && ((_WORD)v18 || v22 == v72)
    || ((v79 = v18,
         v80 = v72,
         v33.m128i_i64[0] = sub_2D5B750(&v79),
         v83 = v33,
         v34 = sub_2D5B750(v77),
         v82 = v35,
         v81 = v34,
         (_BYTE)v35)
     || !v83.m128i_i8[8])
    && v81 >= v83.m128i_i64[0] )
  {
    v23 = *(_QWORD *)(a2 + 80);
    v24 = a1[1];
    v83.m128i_i64[0] = v23;
    if ( v23 )
      sub_B96E90((__int64)&v83, v23, 1);
    v83.m128i_i32[2] = *(_DWORD *)(a2 + 72);
    v25 = sub_33FAF80(v24, v20, (__int64)&v83, *(unsigned int *)v77, v78, v22, a3);
    if ( v83.m128i_i64[0] )
      sub_B91220((__int64)&v83, v83.m128i_i64[0]);
  }
  else
  {
    sub_33FAF80(a1[1], v20, (__int64)&v74, v3, v72, v22, a3);
    v25 = sub_33FAF80(a1[1], 216, (__int64)&v74, *(unsigned int *)v77, v78, v36, a3);
  }
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  return v25;
}
