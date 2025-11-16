// Function: sub_379E370
// Address: 0x379e370
//
unsigned __int8 *__fastcall sub_379E370(__int64 a1, unsigned __int64 a2, int a3, __m128i a4)
{
  __int64 v6; // rsi
  __int64 v7; // r9
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 (__fastcall *v14)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  int v18; // eax
  unsigned __int16 *v19; // rdx
  unsigned int v20; // ebx
  int v21; // eax
  unsigned __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // eax
  unsigned __int16 *v28; // rdx
  unsigned int v29; // ebx
  int v30; // eax
  unsigned __int16 v31; // ax
  __int64 v32; // rdx
  unsigned int v33; // r14d
  __int16 v34; // ax
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned int v37; // edx
  unsigned int v38; // edx
  unsigned int *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r9
  __int64 v42; // rbx
  unsigned __int8 *v43; // r14
  unsigned __int16 *v44; // rax
  __int128 v45; // rax
  unsigned __int8 *v46; // rax
  __int64 v47; // rdx
  unsigned int v49; // r14d
  __int16 v50; // ax
  __int64 v51; // rcx
  __int64 v52; // rdi
  __int128 v53; // rax
  _QWORD *v54; // r14
  __int128 *v55; // rbx
  __int128 v56; // rax
  __int64 v57; // r9
  __int64 v58; // rbx
  _QWORD *v59; // r14
  unsigned int v60; // edx
  __int128 v61; // rax
  __int64 v62; // r9
  unsigned int v63; // edx
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int128 v68; // [rsp-20h] [rbp-150h]
  __int64 v69; // [rsp+0h] [rbp-130h]
  __int64 *v70; // [rsp+0h] [rbp-130h]
  __int128 v71; // [rsp+0h] [rbp-130h]
  __int128 v72; // [rsp+0h] [rbp-130h]
  __int64 v73; // [rsp+10h] [rbp-120h]
  __int64 *v74; // [rsp+10h] [rbp-120h]
  __int128 v75; // [rsp+10h] [rbp-120h]
  __int128 v77; // [rsp+30h] [rbp-100h]
  unsigned __int16 v78; // [rsp+30h] [rbp-100h]
  __int128 v79; // [rsp+40h] [rbp-F0h]
  __int64 v80; // [rsp+40h] [rbp-F0h]
  __int64 v81; // [rsp+60h] [rbp-D0h]
  __int64 v82; // [rsp+90h] [rbp-A0h] BYREF
  int v83; // [rsp+98h] [rbp-98h]
  __int64 v84[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v85; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v86; // [rsp+B8h] [rbp-78h]
  __int64 v87; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v88; // [rsp+C8h] [rbp-68h]
  unsigned int v89; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v90; // [rsp+D8h] [rbp-58h]
  __int64 v91; // [rsp+E0h] [rbp-50h] BYREF
  int v92; // [rsp+E8h] [rbp-48h]
  __int64 v93; // [rsp+F0h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v82 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v82, v6, 1);
  v7 = *(_QWORD *)a1;
  v83 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v88 = 0;
  v11 = *((_QWORD *)v8 + 3);
  LOWORD(v8) = v8[8];
  v90 = 0;
  LOWORD(v84[0]) = v9;
  LOWORD(v85) = (_WORD)v8;
  LOWORD(v87) = 0;
  LOWORD(v89) = 0;
  v84[1] = v10;
  v12 = *(_QWORD *)(a1 + 8);
  v86 = v11;
  v13 = *(_QWORD *)(v12 + 64);
  v14 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  if ( !a3 )
  {
    if ( v14 == sub_2D56A50 )
    {
      v15 = v7;
      sub_2FE6CC0((__int64)&v91, v7, v13, v84[0], v10);
      LOWORD(v18) = v92;
      LOWORD(v87) = v92;
      v88 = v93;
    }
    else
    {
      v15 = v13;
      v18 = v14(v7, v13, v84[0], v10);
      LODWORD(v87) = v18;
      v88 = v64;
    }
    if ( (_WORD)v18 )
    {
      if ( (unsigned __int16)(v18 - 176) > 0x34u )
        goto LABEL_8;
    }
    else if ( !sub_3007100((__int64)&v87) )
    {
      goto LABEL_17;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v87 )
    {
      if ( (unsigned __int16)(v87 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_8:
      v19 = word_4456340;
      v20 = word_4456340[(unsigned __int16)v87 - 1];
      v21 = (unsigned __int16)v85;
      if ( !(_WORD)v85 )
      {
LABEL_9:
        v22 = sub_3009970((__int64)&v85, v15, (__int64)v19, v16, v17);
        v73 = v23;
LABEL_19:
        v33 = v22;
        v70 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
        v34 = sub_2D43050(v22, v20);
        v35 = 0;
        if ( !v34 )
        {
          v34 = sub_3009400(v70, v33, v73, v20, 0);
          v35 = v66;
        }
        LOWORD(v89) = v34;
        v36 = *(_QWORD *)(a2 + 40);
        v90 = v35;
        *(_QWORD *)&v79 = sub_379AB60(a1, *(_QWORD *)v36, *(_QWORD *)(v36 + 8));
        *((_QWORD *)&v79 + 1) = v37;
        *(_QWORD *)&v77 = sub_379AB60(
                            a1,
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
        *((_QWORD *)&v77 + 1) = v38;
        goto LABEL_22;
      }
LABEL_18:
      v73 = 0;
      v22 = word_4456580[v21 - 1];
      goto LABEL_19;
    }
LABEL_17:
    v20 = sub_3007130((__int64)&v87, v15);
    v21 = (unsigned __int16)v85;
    if ( !(_WORD)v85 )
      goto LABEL_9;
    goto LABEL_18;
  }
  if ( v14 == sub_2D56A50 )
  {
    v24 = v7;
    sub_2FE6CC0((__int64)&v91, v7, v13, v85, v86);
    LOWORD(v27) = v92;
    LOWORD(v89) = v92;
    v90 = v93;
  }
  else
  {
    v24 = v13;
    v27 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v14)(v7, v13, (unsigned int)v85);
    v89 = v27;
    v90 = v67;
  }
  if ( (_WORD)v27 )
  {
    if ( (unsigned __int16)(v27 - 176) > 0x34u )
      goto LABEL_14;
  }
  else if ( !sub_3007100((__int64)&v89) )
  {
    goto LABEL_29;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v89 )
  {
LABEL_29:
    v29 = sub_3007130((__int64)&v89, v24);
    v30 = LOWORD(v84[0]);
    if ( !LOWORD(v84[0]) )
      goto LABEL_15;
LABEL_30:
    v69 = 0;
    v31 = word_4456580[v30 - 1];
    goto LABEL_31;
  }
  if ( (unsigned __int16)(v89 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_14:
  v28 = word_4456340;
  v29 = word_4456340[(unsigned __int16)v89 - 1];
  v30 = LOWORD(v84[0]);
  if ( LOWORD(v84[0]) )
    goto LABEL_30;
LABEL_15:
  v31 = sub_3009970((__int64)v84, v24, (__int64)v28, v25, v26);
  v69 = v32;
LABEL_31:
  v49 = v31;
  v74 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  v50 = sub_2D43050(v31, v29);
  v51 = 0;
  if ( !v50 )
  {
    v50 = sub_3009400(v74, v49, v69, v29, 0);
    v51 = v65;
  }
  v52 = *(_QWORD *)(a1 + 8);
  v88 = v51;
  LOWORD(v87) = v50;
  *(_QWORD *)&v53 = sub_3400EE0(v52, 0, (__int64)&v82, 0, a4);
  v54 = *(_QWORD **)(a1 + 8);
  v75 = v53;
  v91 = 0;
  v55 = *(__int128 **)(a2 + 40);
  v92 = 0;
  *(_QWORD *)&v56 = sub_33F17F0(v54, 51, (__int64)&v91, v87, v88);
  if ( v91 )
  {
    v71 = v56;
    sub_B91220((__int64)&v91, v91);
    v56 = v71;
  }
  v81 = sub_340F900(v54, 0xA0u, (__int64)&v82, v87, v88, v57, v56, *v55, v75);
  v58 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)&v79 = v81;
  v59 = *(_QWORD **)(a1 + 8);
  v91 = 0;
  v92 = 0;
  *((_QWORD *)&v79 + 1) = v60;
  *(_QWORD *)&v61 = sub_33F17F0(v59, 51, (__int64)&v91, v87, v88);
  if ( v91 )
  {
    v72 = v61;
    sub_B91220((__int64)&v91, v91);
    v61 = v72;
  }
  *(_QWORD *)&v77 = sub_340F900(v59, 0xA0u, (__int64)&v82, v87, v88, v62, v61, *(_OWORD *)(v58 + 40), v75);
  *((_QWORD *)&v77 + 1) = v63;
LABEL_22:
  v39 = (unsigned int *)sub_33E5110(*(__int64 **)(a1 + 8), (unsigned int)v87, v88, v89, v90);
  v42 = (unsigned int)(1 - a3);
  v43 = sub_3411F20(*(_QWORD **)(a1 + 8), *(unsigned int *)(a2 + 24), (__int64)&v82, v39, v40, v41, v79, v77);
  v44 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v42);
  v78 = *v44;
  v80 = *((_QWORD *)v44 + 1);
  sub_2FE6CC0((__int64)&v91, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), *v44, v80);
  if ( (_BYTE)v91 == 7 )
  {
    sub_3760B50(a1, a2, v42, (unsigned __int64)v43, v42);
  }
  else
  {
    *(_QWORD *)&v45 = sub_3400EE0(*(_QWORD *)(a1 + 8), 0, (__int64)&v82, 0, a4);
    *((_QWORD *)&v68 + 1) = (unsigned int)(1 - a3);
    *(_QWORD *)&v68 = v43;
    v46 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0xA1u, (__int64)&v82, v78, v80, *(_QWORD *)(a1 + 8), v68, v45);
    sub_3760E70(a1, a2, v42, (unsigned __int64)v46, v47);
  }
  if ( v82 )
    sub_B91220((__int64)&v82, v82);
  return v43;
}
