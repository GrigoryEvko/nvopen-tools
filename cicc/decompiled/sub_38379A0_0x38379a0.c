// Function: sub_38379A0
// Address: 0x38379a0
//
__int64 __fastcall sub_38379A0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r12
  const __m128i *v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // r13
  __int64 v10; // rsi
  unsigned __int16 *v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r13d
  unsigned __int16 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int16 *v17; // r12
  __int64 v18; // rsi
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int16 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r11
  __int64 v25; // rdx
  __int128 v26; // rax
  unsigned __int8 *v27; // rax
  unsigned int v28; // r8d
  unsigned int v29; // edx
  int v30; // eax
  __int128 v31; // rax
  __int64 v32; // r15
  __int64 v33; // r14
  __int64 v34; // r9
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // r14
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // r8
  unsigned __int32 v45; // edx
  __int128 v46; // rax
  __int64 v47; // r9
  unsigned int v48; // edx
  unsigned int v49; // edx
  __int64 v50; // r9
  __int128 v51; // rax
  __int64 v52; // r9
  _QWORD *v53; // rdi
  unsigned int v54; // edx
  __int64 v55; // r9
  unsigned __int8 *v56; // rax
  unsigned int v57; // edx
  __int16 v58; // ax
  __int64 v59; // r8
  __int128 v60; // [rsp-20h] [rbp-180h]
  __int128 v61; // [rsp-10h] [rbp-170h]
  __int128 v62; // [rsp-10h] [rbp-170h]
  __int128 v63; // [rsp-10h] [rbp-170h]
  unsigned int v64; // [rsp+0h] [rbp-160h]
  _QWORD *v65; // [rsp+8h] [rbp-158h]
  __int128 v66; // [rsp+10h] [rbp-150h]
  int v67; // [rsp+20h] [rbp-140h]
  unsigned __int32 v68; // [rsp+24h] [rbp-13Ch]
  unsigned int v69; // [rsp+24h] [rbp-13Ch]
  __int64 v70; // [rsp+28h] [rbp-138h]
  unsigned int v71; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v72; // [rsp+38h] [rbp-128h]
  unsigned __int8 *v73; // [rsp+38h] [rbp-128h]
  __int64 v74; // [rsp+40h] [rbp-120h]
  __int128 v75; // [rsp+40h] [rbp-120h]
  __int128 v76; // [rsp+50h] [rbp-110h]
  __int128 v78; // [rsp+60h] [rbp-100h]
  unsigned __int64 v79; // [rsp+68h] [rbp-F8h]
  __int64 v80; // [rsp+68h] [rbp-F8h]
  __int64 v81; // [rsp+C0h] [rbp-A0h] BYREF
  int v82; // [rsp+C8h] [rbp-98h]
  unsigned int v83; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v84; // [rsp+D8h] [rbp-88h]
  unsigned int v85; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v86; // [rsp+E8h] [rbp-78h]
  unsigned __int16 v87; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v88; // [rsp+F8h] [rbp-68h]
  __int64 v89; // [rsp+100h] [rbp-60h]
  __int64 v90; // [rsp+108h] [rbp-58h]
  __int64 v91; // [rsp+110h] [rbp-50h] BYREF
  __int64 v92; // [rsp+118h] [rbp-48h]

  *(_QWORD *)&v66 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v66 + 1) = v4;
  *(_QWORD *)&v76 = sub_37AE0F0(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v6 = (unsigned int)v5;
  v7 = *(const __m128i **)(a2 + 40);
  *((_QWORD *)&v76 + 1) = v5;
  v8 = _mm_loadu_si128(v7 + 5);
  v68 = v7[5].m128i_u32[2];
  v9 = 16LL * v68;
  v72 = (unsigned __int8 *)v7[5].m128i_i64[0];
  sub_2FE6CC0(
    (__int64)&v91,
    *a1,
    *(_QWORD *)(a1[1] + 64),
    *(unsigned __int16 *)(v9 + *((_QWORD *)v72 + 6)),
    *(_QWORD *)(v9 + *((_QWORD *)v72 + 6) + 8));
  if ( (_BYTE)v91 == 1 )
  {
    v72 = sub_37AF270((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1], v8);
    v68 = v45;
    v9 = 16LL * v45;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = (unsigned __int16 *)(v9 + *((_QWORD *)v72 + 6));
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  v81 = v10;
  v74 = v12;
  if ( v10 )
    sub_B96E90((__int64)&v81, v10, 1);
  v82 = *(_DWORD *)(a2 + 72);
  v14 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  LODWORD(v15) = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v84 = v16;
  v17 = (unsigned __int16 *)(*(_QWORD *)(v76 + 48) + 16 * v6);
  LOWORD(v83) = v15;
  v18 = *v17;
  v86 = *((_QWORD *)v17 + 1);
  LODWORD(v14) = *(_DWORD *)(a2 + 24);
  LOWORD(v85) = v18;
  v71 = (unsigned int)v14;
  if ( (_WORD)v15 )
  {
    if ( (unsigned __int16)(v15 - 17) > 0xD3u )
    {
      LOWORD(v91) = v15;
      v92 = v16;
      goto LABEL_8;
    }
    LOWORD(v15) = word_4456580[(int)v15 - 1];
    v20 = 0;
  }
  else
  {
    v70 = v16;
    if ( !sub_30070B0((__int64)&v83) )
    {
      v92 = v70;
      LOWORD(v91) = 0;
      goto LABEL_13;
    }
    v58 = sub_3009970((__int64)&v83, v18, v43, v70, v44);
    v59 = v15;
    LOWORD(v15) = v58;
    v20 = v59;
  }
  LOWORD(v91) = v15;
  v92 = v20;
  if ( !(_WORD)v15 )
  {
LABEL_13:
    v89 = sub_3007260((__int64)&v91);
    LODWORD(v19) = v89;
    v90 = v21;
    goto LABEL_14;
  }
LABEL_8:
  if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
    goto LABEL_42;
  v19 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16];
LABEL_14:
  v22 = v85;
  if ( !(_WORD)v85 )
  {
    if ( sub_30070B0((__int64)&v85) )
    {
      v22 = sub_3009970((__int64)&v85, v18, v39, v40, v41);
      v88 = v42;
      v87 = v22;
      if ( !v22 )
        goto LABEL_18;
      goto LABEL_31;
    }
    goto LABEL_16;
  }
  if ( (unsigned __int16)(v85 - 17) > 0xD3u )
  {
LABEL_16:
    v23 = v86;
    goto LABEL_17;
  }
  v23 = 0;
  v22 = word_4456580[(unsigned __int16)v85 - 1];
LABEL_17:
  v87 = v22;
  v88 = v23;
  if ( v22 )
  {
LABEL_31:
    if ( v22 != 1 && (unsigned __int16)(v22 - 504) > 7u )
    {
      v24 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
      goto LABEL_19;
    }
LABEL_42:
    BUG();
  }
LABEL_18:
  v91 = sub_3007260((__int64)&v87);
  LODWORD(v24) = v91;
  v92 = v25;
LABEL_19:
  v67 = v24;
  v64 = v24;
  v65 = (_QWORD *)a1[1];
  *(_QWORD *)&v26 = sub_3400BD0((__int64)v65, (unsigned int)v19, (__int64)&v81, v13, v74, 0, v8, 0);
  v79 = v68 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v27 = sub_3406EB0(
          v65,
          0x3Eu,
          (__int64)&v81,
          v13,
          v74,
          *((__int64 *)&v26 + 1),
          __PAIR128__(v79, (unsigned __int64)v72),
          v26);
  v69 = v29;
  v73 = v27;
  *(_QWORD *)&v78 = v27;
  *((_QWORD *)&v78 + 1) = v29 | v79 & 0xFFFFFFFF00000000LL;
  if ( 2 * (int)v19 > v64
    || (v30 = *((_DWORD *)v27 + 6), v30 == 35)
    || v30 == 11
    || (unsigned __int8)sub_3813820(*a1, v71, v85, 0, v28) )
  {
    *(_QWORD *)&v31 = sub_3400BD0(a1[1], (unsigned int)(v67 - v19), (__int64)&v81, v13, v74, 0, v8, 0);
    v32 = *((_QWORD *)&v31 + 1);
    v33 = v31;
    *(_QWORD *)&v76 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v81, v85, v86, v34, v76, v31);
    *((_QWORD *)&v76 + 1) = v36 | *((_QWORD *)&v76 + 1) & 0xFFFFFFFF00000000LL;
    if ( v71 == 196 )
    {
      *((_QWORD *)&v63 + 1) = v32;
      *(_QWORD *)&v63 = v33;
      v73 = sub_3406EB0((_QWORD *)a1[1], 0x38u, (__int64)&v81, v13, v74, v35, v78, v63);
      v69 = v57;
    }
    v37 = sub_340F900(
            (_QWORD *)a1[1],
            v71,
            (__int64)&v81,
            v85,
            v86,
            v35,
            v66,
            v76,
            __PAIR128__(v69 | *((_QWORD *)&v78 + 1) & 0xFFFFFFFF00000000LL, (unsigned __int64)v73));
  }
  else
  {
    *(_QWORD *)&v46 = sub_3400BD0(a1[1], (unsigned int)v19, (__int64)&v81, v85, v86, 0, v8, 0);
    v75 = v46;
    *(_QWORD *)&v66 = sub_3406EB0((_QWORD *)a1[1], 0xBEu, (__int64)&v81, v85, v86, v47, v66, v46);
    *((_QWORD *)&v66 + 1) = v48 | *((_QWORD *)&v66 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v76 = sub_34070B0((_QWORD *)a1[1], v76, *((__int64 *)&v76 + 1), (__int64)&v81, v83, v84, v8);
    *((_QWORD *)&v61 + 1) = v49 | *((_QWORD *)&v76 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v61 = v76;
    *(_QWORD *)&v51 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, (__int64)&v81, v85, v86, v50, v66, v61);
    v53 = (_QWORD *)a1[1];
    if ( v71 == 196 )
    {
      v56 = sub_3406EB0(v53, 0xC0u, (__int64)&v81, v85, v86, v52, v51, v78);
    }
    else
    {
      v62 = v78;
      v80 = *((_QWORD *)&v51 + 1);
      *(_QWORD *)&v60 = sub_3406EB0(v53, 0xBEu, (__int64)&v81, v85, v86, v52, v51, v62);
      *((_QWORD *)&v60 + 1) = v54 | v80 & 0xFFFFFFFF00000000LL;
      v56 = sub_3406EB0((_QWORD *)a1[1], 0xC0u, (__int64)&v81, v85, v86, v55, v60, v75);
    }
    v37 = (__int64)v56;
  }
  if ( v81 )
    sub_B91220((__int64)&v81, v81);
  return v37;
}
