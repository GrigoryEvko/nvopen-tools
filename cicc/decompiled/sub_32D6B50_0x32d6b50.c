// Function: sub_32D6B50
// Address: 0x32d6b50
//
__int64 __fastcall sub_32D6B50(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int16 *v4; // rdx
  __int64 v5; // r12
  const __m128i *v6; // roff
  _QWORD *v7; // r13
  __int64 v8; // r14
  __int32 v9; // eax
  int v10; // eax
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r12
  __int32 v16; // r9d
  unsigned __int16 *v17; // rdx
  int v18; // eax
  __int64 v19; // r15
  bool v20; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdx
  char v26; // al
  char v27; // al
  char v28; // cl
  unsigned __int16 *v29; // rax
  __m128i v30; // rax
  __int64 v31; // rdi
  __int128 v32; // rax
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // r9d
  __int64 v37; // rcx
  __int64 v38; // rax
  __m128i v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // rcx
  unsigned int v43; // edx
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  unsigned int v47; // edx
  __int128 v48; // rax
  char v49; // al
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int16 v52; // cx
  __int64 v53; // r8
  __int64 v54; // rdi
  __m128i v55; // rax
  __m128i v56; // xmm1
  __int64 v57; // rdi
  __int64 v58; // r12
  __int64 v59; // rdx
  __int64 v60; // r13
  __m128i v61; // xmm3
  __int64 v62; // rdi
  __m128i v63; // rax
  __int64 v64; // rdi
  __m128i v65; // rax
  __int64 v66; // r12
  __int64 *v67; // r13
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rdi
  __m128i v71; // xmm5
  __int128 v72; // rax
  int v73; // r9d
  __int64 v74; // rdx
  __int64 v75; // r8
  int v76; // [rsp+8h] [rbp-138h]
  __int32 v77; // [rsp+10h] [rbp-130h]
  __int64 v78; // [rsp+10h] [rbp-130h]
  __int64 v79; // [rsp+10h] [rbp-130h]
  unsigned int v80; // [rsp+10h] [rbp-130h]
  __int32 v81; // [rsp+10h] [rbp-130h]
  __int32 v82; // [rsp+18h] [rbp-128h]
  char v83; // [rsp+18h] [rbp-128h]
  int v84; // [rsp+18h] [rbp-128h]
  int v85; // [rsp+18h] [rbp-128h]
  int v86; // [rsp+18h] [rbp-128h]
  __int64 v87; // [rsp+20h] [rbp-120h]
  unsigned __int32 v88; // [rsp+28h] [rbp-118h]
  int v89; // [rsp+28h] [rbp-118h]
  __m128i v90; // [rsp+30h] [rbp-110h] BYREF
  _QWORD *v91; // [rsp+40h] [rbp-100h]
  _QWORD *v92; // [rsp+48h] [rbp-F8h]
  __int64 v93; // [rsp+50h] [rbp-F0h]
  __int64 v94; // [rsp+58h] [rbp-E8h]
  char v95; // [rsp+6Fh] [rbp-D1h] BYREF
  __int64 v96; // [rsp+70h] [rbp-D0h] BYREF
  int v97; // [rsp+78h] [rbp-C8h]
  int v98; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+88h] [rbp-B8h]
  __int16 v100; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+98h] [rbp-A8h]
  __int64 v102; // [rsp+A0h] [rbp-A0h]
  __int64 v103; // [rsp+A8h] [rbp-98h]
  __int16 v104; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v105; // [rsp+B8h] [rbp-88h]
  __int64 v106; // [rsp+C0h] [rbp-80h]
  __int64 v107; // [rsp+C8h] [rbp-78h]
  __m128i v108; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v109; // [rsp+E0h] [rbp-60h]
  __m128i v110; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v111; // [rsp+100h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v92 = a1;
  v96 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v96, v3, 1);
  v4 = *(unsigned __int16 **)(a2 + 48);
  v97 = *(_DWORD *)(a2 + 72);
  v5 = *((_QWORD *)v4 + 1);
  v6 = *(const __m128i **)(a2 + 40);
  v7 = (_QWORD *)v6[2].m128i_i64[1];
  v8 = v6[3].m128i_i64[0];
  v9 = v6[3].m128i_i32[0];
  v87 = v6->m128i_i64[0];
  v91 = v7;
  v88 = v9;
  v10 = *v4;
  v90 = _mm_loadu_si128(v6);
  LOWORD(v98) = v10;
  v99 = v5;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    {
      v100 = v10;
      v101 = v5;
      goto LABEL_6;
    }
    LOWORD(v10) = word_4456580[v10 - 1];
    v12 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v98) )
    {
      v101 = v5;
      v100 = 0;
      goto LABEL_11;
    }
    LOWORD(v10) = sub_3009970((__int64)&v98, v3, v44, v45, v46);
  }
  v100 = v10;
  v101 = v12;
  if ( !(_WORD)v10 )
  {
LABEL_11:
    v102 = sub_3007260((__int64)&v100);
    LODWORD(v11) = v102;
    v103 = v13;
    goto LABEL_12;
  }
LABEL_6:
  if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
    goto LABEL_102;
  v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
LABEL_12:
  if ( (unsigned __int8)sub_33E0720(v7, v8, 0) )
  {
    v14 = v90.m128i_i64[0];
    goto LABEL_14;
  }
  v16 = v11;
  if ( !(_DWORD)v11 || ((unsigned int)v11 & ((_DWORD)v11 - 1)) != 0 || (_DWORD)v11 == 1 )
  {
LABEL_33:
    v108.m128i_i32[0] = v16;
    v108.m128i_i64[1] = (__int64)&v95;
    v109.m128i_i64[1] = (__int64)sub_3261AE0;
    v95 = 0;
    v109.m128i_i64[0] = (__int64)sub_325DD40;
    v111.m128i_i64[0] = 0;
    sub_325DD40((const __m128i **)&v110, &v108, 2);
    v111 = v109;
    v27 = sub_33CA8D0(v7, v8, &v110);
    v28 = v27;
    if ( v111.m128i_i64[0] )
    {
      v83 = v27;
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v111.m128i_i64[0])(&v110, &v110, 3);
      v28 = v83;
    }
    if ( v28 && v95 )
    {
      if ( v109.m128i_i64[0] )
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v109.m128i_i64[0])(&v108, &v108, 3);
      v29 = (unsigned __int16 *)(v91[6] + 16LL * v88);
      v76 = *v29;
      v78 = *((_QWORD *)v29 + 1);
      v30.m128i_i64[0] = sub_3400BD0(*v92, v11, (unsigned int)&v96, v76, v78, 0, 0);
      v31 = *v92;
      v110.m128i_i64[0] = (__int64)v7;
      v110.m128i_i64[1] = v8;
      v111 = v30;
      *(_QWORD *)&v32 = sub_3402EA0(v31, 62, (unsigned int)&v96, v76, v78, 0, (__int64)&v110, 2);
      if ( (_QWORD)v32 )
      {
        v34 = sub_3406EB0(*v92, *(_DWORD *)(a2 + 24), (unsigned int)&v96, v98, v99, v33, *(_OWORD *)&v90, v32);
        goto LABEL_41;
      }
    }
    else if ( v109.m128i_i64[0] )
    {
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v109.m128i_i64[0])(&v108, &v108, 3);
    }
    v35 = sub_33DFBC0(v7, v8, 0, 0);
    if ( !v35 )
      goto LABEL_67;
    v37 = *(_QWORD *)(v35 + 96);
    if ( *(_DWORD *)(v37 + 32) <= 0x40u )
    {
      v38 = *(_QWORD *)(v37 + 24);
    }
    else
    {
      v79 = *(_QWORD *)(v35 + 96);
      v84 = *(_DWORD *)(v37 + 32);
      if ( v84 - (unsigned int)sub_C444A0(v37 + 24) > 0x40 )
        goto LABEL_67;
      v38 = **(_QWORD **)(v79 + 24);
    }
    if ( v38 != 8 )
      goto LABEL_67;
    v39.m128i_i16[4] = v98;
    if ( (_WORD)v98 )
    {
      if ( (unsigned __int16)(v98 - 17) <= 0xD3u )
      {
        v39.m128i_i16[4] = word_4456580[(unsigned __int16)v98 - 1];
        v39.m128i_i64[0] = 0;
        goto LABEL_52;
      }
    }
    else
    {
      v80 = (unsigned __int16)v98;
      v39.m128i_i8[0] = sub_30070B0((__int64)&v98);
      v39.m128i_i16[4] = v80;
      if ( v39.m128i_i8[0] )
      {
        v39.m128i_i16[0] = sub_3009970((__int64)&v98, v8, v80, v40, v41);
        v75 = v39.m128i_i64[1];
        v39.m128i_i16[4] = v39.m128i_i16[0];
        v39.m128i_i64[0] = v75;
        goto LABEL_52;
      }
    }
    v39.m128i_i64[0] = v99;
LABEL_52:
    v110.m128i_i16[0] = v39.m128i_i16[4];
    v110.m128i_i64[1] = v39.m128i_i64[0];
    if ( !v39.m128i_i16[4] )
    {
      v39.m128i_i64[0] = sub_3007260((__int64)&v110);
      v108 = v39;
      goto LABEL_54;
    }
    if ( v39.m128i_i16[4] != 1 && (unsigned __int16)(v39.m128i_i16[4] - 504) > 7u )
    {
      v39.m128i_i64[0] = *(_QWORD *)&byte_444C4A0[16 * v39.m128i_u16[4] - 16];
LABEL_54:
      if ( v39.m128i_i64[0] == 16 )
      {
        v42 = v92[1];
        if ( *((_BYTE *)v92 + 33) )
        {
          v43 = 1;
          if ( (_WORD)v98 == 1
            || (_WORD)v98 && (v43 = (unsigned __int16)v98, *(_QWORD *)(v42 + 8LL * (unsigned __int16)v98 + 112)) )
          {
            if ( !*(_BYTE *)(v42 + 500LL * v43 + 6611) )
            {
LABEL_60:
              v14 = sub_33FAF80(*v92, 197, (unsigned int)&v96, v98, v99, v36, *(_OWORD *)&v90);
              goto LABEL_14;
            }
          }
        }
        else
        {
          v47 = 1;
          if ( (_WORD)v98 == 1
            || (_WORD)v98 && (v47 = (unsigned __int16)v98, *(_QWORD *)(v42 + 8LL * (unsigned __int16)v98 + 112)) )
          {
            if ( (*(_BYTE *)(v42 + 500LL * v47 + 6611) & 0xFB) == 0 )
              goto LABEL_60;
          }
        }
      }
LABEL_67:
      if ( (unsigned __int8)sub_32D0FE0((__int64)v92, a2, 0) )
      {
        v14 = a2;
        goto LABEL_14;
      }
      if ( *((_DWORD *)v91 + 6) != 216
        || *(_DWORD *)(*(_QWORD *)v91[5] + 24LL) != 186
        || (*(_QWORD *)&v48 = sub_32CB9C0((__int64)v92, v91), !(_QWORD)v48) )
      {
        if ( (unsigned int)(*(_DWORD *)(v87 + 24) - 193) > 1 )
          goto LABEL_70;
        v85 = *(_DWORD *)(v87 + 24);
        v90.m128i_i8[0] = sub_33E2390(*v92, v7, v8, 1);
        v49 = sub_33E2390(*v92, *(_QWORD *)(*(_QWORD *)(v87 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v87 + 40) + 48LL), 1);
        if ( !v90.m128i_i8[0] )
          goto LABEL_70;
        if ( !v49 )
          goto LABEL_70;
        v50 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v87 + 40) + 40LL) + 48LL)
            + 16LL * *(unsigned int *)(*(_QWORD *)(v87 + 40) + 48LL);
        v51 = v91[6] + 16LL * v88;
        v52 = *(_WORD *)v51;
        if ( *(_WORD *)v51 != *(_WORD *)v50 )
          goto LABEL_70;
        v53 = *(_QWORD *)(v51 + 8);
        if ( *(_QWORD *)(v50 + 8) != v53 && !v52 )
          goto LABEL_70;
        v86 = (v85 != *(_DWORD *)(a2 + 24)) + 56;
        v54 = *v92;
        v89 = v52;
        v91 = *(_QWORD **)(v51 + 8);
        v55.m128i_i64[0] = sub_3400BD0(v54, v11, (unsigned int)&v96, v52, v53, 0, 0);
        v90 = v55;
        v56 = _mm_load_si128(&v90);
        v57 = *v92;
        v110.m128i_i64[0] = (__int64)v7;
        v110.m128i_i64[1] = v8;
        v111 = v56;
        v58 = sub_3402EA0(v57, 62, (unsigned int)&v96, v89, (_DWORD)v91, 0, (__int64)&v110, 2);
        v60 = v59;
        v61 = _mm_load_si128(&v90);
        v62 = *v92;
        v110 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v87 + 40) + 40LL));
        v111 = v61;
        v63.m128i_i64[0] = sub_3402EA0(v62, 62, (unsigned int)&v96, v89, (_DWORD)v91, 0, (__int64)&v110, 2);
        if ( !v58 )
          goto LABEL_70;
        if ( v63.m128i_i64[0]
          && (v110.m128i_i64[0] = v58,
              v64 = *v92,
              v110.m128i_i64[1] = v60,
              v111 = v63,
              v65.m128i_i64[0] = sub_3402EA0(v64, v86, (unsigned int)&v96, v89, (_DWORD)v91, 0, (__int64)&v110, 2),
              v66 = v65.m128i_i64[1],
              v65.m128i_i64[0]) )
        {
          v67 = v92;
          v110 = v65;
          v68 = *v92;
          v111 = _mm_load_si128(&v90);
          v92 = v91;
          v93 = sub_3402EA0(v68, 56, (unsigned int)&v96, v89, (_DWORD)v91, 0, (__int64)&v110, 2);
          v94 = v69;
          v70 = *v67;
          v71 = _mm_load_si128(&v90);
          v110.m128i_i64[0] = v93;
          v111 = v71;
          v110.m128i_i64[1] = (unsigned int)v69 | v66 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v72 = sub_3402EA0(v70, 62, (unsigned int)&v96, v89, (_DWORD)v92, 0, (__int64)&v110, 2);
          v14 = sub_3406EB0(
                  *v67,
                  *(_DWORD *)(a2 + 24),
                  (unsigned int)&v96,
                  v98,
                  v99,
                  v73,
                  *(_OWORD *)*(_QWORD *)(v87 + 40),
                  v72);
        }
        else
        {
LABEL_70:
          v14 = 0;
        }
        goto LABEL_14;
      }
      v34 = sub_3406EB0(
              *v92,
              *(_DWORD *)(a2 + 24),
              (unsigned int)&v96,
              v98,
              v99,
              (unsigned int)&v96,
              *(_OWORD *)&v90,
              v48);
LABEL_41:
      v14 = v34;
      goto LABEL_14;
    }
LABEL_102:
    BUG();
  }
  v17 = (unsigned __int16 *)(v91[6] + 16LL * v88);
  v18 = *v17;
  v19 = *((_QWORD *)v17 + 1);
  v104 = v18;
  v105 = v19;
  if ( (_WORD)v18 )
  {
    if ( (unsigned __int16)(v18 - 17) > 0xD3u )
    {
      v108.m128i_i16[0] = v18;
      v108.m128i_i64[1] = v19;
      goto LABEL_81;
    }
    LOWORD(v18) = word_4456580[v18 - 1];
    v74 = 0;
  }
  else
  {
    v20 = sub_30070B0((__int64)&v104);
    v16 = v11;
    if ( !v20 )
    {
      v108.m128i_i64[1] = v19;
      v108.m128i_i16[0] = 0;
LABEL_23:
      v77 = v16;
      v24 = sub_3007260((__int64)&v108);
      v16 = v77;
      v106 = v24;
      v107 = v25;
      goto LABEL_24;
    }
    LOWORD(v18) = sub_3009970((__int64)&v104, v8, v21, v22, v23);
    v16 = v11;
  }
  v108.m128i_i16[0] = v18;
  v108.m128i_i64[1] = v74;
  if ( !(_WORD)v18 )
    goto LABEL_23;
LABEL_81:
  if ( (_WORD)v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
    goto LABEL_102;
  v24 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v18 - 16];
LABEL_24:
  v110.m128i_i32[2] = v24;
  if ( (unsigned int)v24 > 0x40 )
  {
    v81 = v16;
    sub_C43690((__int64)&v110, (unsigned int)(v11 - 1), 0);
    v16 = v81;
  }
  else
  {
    v110.m128i_i64[0] = (unsigned int)(v11 - 1);
  }
  v82 = v16;
  v26 = sub_33DD210(*v92, v7, v8, &v110, 0);
  v16 = v82;
  if ( !v26 )
  {
    if ( v110.m128i_i32[2] > 0x40u && v110.m128i_i64[0] )
    {
      j_j___libc_free_0_0(v110.m128i_u64[0]);
      v16 = v82;
    }
    goto LABEL_33;
  }
  v14 = v90.m128i_i64[0];
  if ( v110.m128i_i32[2] > 0x40u && v110.m128i_i64[0] )
    j_j___libc_free_0_0(v110.m128i_u64[0]);
LABEL_14:
  if ( v96 )
    sub_B91220((__int64)&v96, v96);
  return v14;
}
