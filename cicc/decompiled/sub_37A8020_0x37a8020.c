// Function: sub_37A8020
// Address: 0x37a8020
//
__int64 __fastcall sub_37A8020(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int16 *v4; // rax
  __int16 v5; // dx
  __int64 *v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int16 v11; // bx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r9
  __int64 v16; // r12
  unsigned __int16 v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v20; // rdx
  char v21; // r8
  __int16 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // r12
  __int64 (__fastcall *v32)(__int64, __int64, __int64, __int64); // r15
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 (__fastcall *v35)(__int64, __int64, unsigned int); // rax
  __int64 v36; // rsi
  int v37; // edx
  unsigned __int16 v38; // ax
  unsigned int v39; // eax
  __int64 v40; // rbx
  int v41; // eax
  __int64 v42; // rdx
  unsigned __int8 *v43; // r14
  __int64 v44; // rdx
  __int64 v45; // r15
  __int64 v46; // rsi
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned int v49; // edx
  _QWORD *v50; // r14
  unsigned __int8 *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r9
  unsigned __int8 *v54; // r8
  __int64 v55; // rsi
  __int64 v56; // r12
  __int64 v57; // rax
  unsigned __int64 v58; // r12
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rsi
  unsigned __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // r12
  unsigned int v66; // edx
  __int64 v67; // rcx
  __int64 v68; // rdx
  _QWORD *v69; // rax
  __int128 v70; // [rsp-30h] [rbp-160h]
  __int128 v71; // [rsp-10h] [rbp-140h]
  __int64 v72; // [rsp-8h] [rbp-138h]
  __int64 v73; // [rsp+0h] [rbp-130h]
  char v74; // [rsp+8h] [rbp-128h]
  __int64 v75; // [rsp+8h] [rbp-128h]
  int v76; // [rsp+10h] [rbp-120h]
  __int64 v77; // [rsp+10h] [rbp-120h]
  __int64 v78; // [rsp+18h] [rbp-118h]
  unsigned int v79; // [rsp+20h] [rbp-110h]
  _QWORD *v80; // [rsp+20h] [rbp-110h]
  unsigned __int8 *v81; // [rsp+20h] [rbp-110h]
  __int64 v82; // [rsp+28h] [rbp-108h]
  __int64 v83; // [rsp+30h] [rbp-100h]
  unsigned int v84; // [rsp+30h] [rbp-100h]
  __int128 v86; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v87; // [rsp+58h] [rbp-D8h]
  __int64 v88; // [rsp+78h] [rbp-B8h] BYREF
  unsigned int v89; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v90; // [rsp+88h] [rbp-A8h]
  unsigned __int16 v91; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v92; // [rsp+98h] [rbp-98h]
  __m128i v93; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v94; // [rsp+B0h] [rbp-80h]
  __int64 v95; // [rsp+B8h] [rbp-78h]
  __m128i v96; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v97; // [rsp+D0h] [rbp-60h]
  __int64 v98; // [rsp+D8h] [rbp-58h]
  _QWORD v99[10]; // [rsp+E0h] [rbp-50h] BYREF

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v90 = *((_QWORD *)v4 + 1);
  v6 = *(__int64 **)(a2 + 40);
  LOWORD(v89) = v5;
  v7 = *a1;
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 5));
  v78 = v6[5];
  v79 = *((_DWORD *)v6 + 12);
  v83 = *v6;
  v87 = v8.m128i_u64[1];
  v9 = 16LL * v79;
  v86 = (__int128)_mm_loadu_si128((const __m128i *)v6);
  v10 = v9 + *(_QWORD *)(v78 + 48);
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v13 = a1[1];
  v91 = v11;
  v14 = *(_QWORD *)(v13 + 64);
  v92 = v12;
  sub_2FE6CC0((__int64)v99, v7, v14, v11, v12);
  if ( LOBYTE(v99[0]) == 7 )
  {
    v78 = sub_379AB60((__int64)a1, v8.m128i_u64[0], v8.m128i_i64[1]);
    v79 = v66;
    v9 = 16LL * v66;
    v87 = v66 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  v16 = *(_QWORD *)(v78 + 48) + v9;
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v93.m128i_i16[0] = v17;
  v93.m128i_i64[1] = v18;
  v96 = _mm_loadu_si128(&v93);
  if ( v17 )
  {
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      goto LABEL_75;
    v19 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
    v21 = byte_444C4A0[16 * v17 - 8];
  }
  else
  {
    v99[0] = sub_3007260((__int64)&v96);
    v19 = v99[0];
    v99[1] = v20;
    v21 = v20;
  }
  v22 = v89;
  if ( !(_WORD)v89 )
  {
    v74 = v21;
    v23 = sub_3007260((__int64)&v89);
    v21 = v74;
    v24 = v23;
    v26 = v25;
    v97 = v24;
    v27 = v24;
    v98 = v26;
    goto LABEL_7;
  }
  if ( (_WORD)v89 == 1 || (unsigned __int16)(v89 - 504) <= 7u )
LABEL_75:
    BUG();
  v27 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v89 - 16];
  LOBYTE(v26) = byte_444C4A0[16 * (unsigned __int16)v89 - 8];
LABEL_7:
  if ( ((_BYTE)v26 || !v21) && v19 <= v27 )
  {
    v28 = *(_QWORD *)(a2 + 80);
    v96.m128i_i64[0] = v28;
    if ( v28 )
      sub_B96E90((__int64)&v96, v28, 1);
    v96.m128i_i32[2] = *(_DWORD *)(a2 + 72);
    goto LABEL_13;
  }
  if ( v22 )
  {
    if ( (unsigned __int16)(v22 - 176) > 0x34u )
    {
LABEL_42:
      v55 = *(_QWORD *)(a2 + 80);
      v96.m128i_i64[0] = v55;
      if ( v55 )
        sub_B96E90((__int64)&v96, v55, 1);
      v96.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      goto LABEL_45;
    }
  }
  else if ( !sub_3007100((__int64)&v89) )
  {
    goto LABEL_42;
  }
  if ( v17 )
  {
    if ( (unsigned __int16)(v17 - 17) > 0x9Eu )
      goto LABEL_42;
  }
  else if ( !sub_30070D0((__int64)&v93) )
  {
    goto LABEL_42;
  }
  v88 = sub_B2D7D0(**(_QWORD **)(a1[1] + 40), 96);
  if ( !v88 )
    goto LABEL_42;
  v56 = (unsigned int)sub_A71EB0(&v88);
  v57 = sub_2D5B750((unsigned __int16 *)&v89);
  v58 = v57 * v56;
  v94 = v57;
  v95 = v59;
  v60 = sub_2D5B750((unsigned __int16 *)&v93);
  v61 = *(_QWORD *)(a2 + 80);
  v62 = v60;
  v96.m128i_i64[1] = v63;
  v96.m128i_i64[0] = v61;
  if ( v61 )
  {
    v77 = v60;
    sub_B96E90((__int64)&v96, v61, 1);
    v62 = v77;
  }
  v96.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  if ( v58 < v62 )
    goto LABEL_45;
LABEL_13:
  if ( *(_DWORD *)(v83 + 24) == 51 )
  {
    v67 = *(_QWORD *)(a2 + 40);
    v68 = *(_QWORD *)(*(_QWORD *)(v67 + 80) + 96LL);
    v69 = *(_QWORD **)(v68 + 24);
    if ( *(_DWORD *)(v68 + 32) > 0x40u )
      v69 = (_QWORD *)*v69;
    if ( !v69 )
    {
      v64 = sub_340F900(
              (_QWORD *)a1[1],
              0xA0u,
              (__int64)&v96,
              v89,
              v90,
              v15,
              v86,
              __PAIR128__(v79 | v87 & 0xFFFFFFFF00000000LL, v78),
              *(_OWORD *)(v67 + 80));
      goto LABEL_55;
    }
  }
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 176) > 0x34u )
      goto LABEL_16;
LABEL_45:
    sub_C64ED0("Don't know how to widen the operands for INSERT_SUBVECTOR", 1u);
  }
  if ( sub_3007100((__int64)&v91) )
    goto LABEL_45;
LABEL_16:
  v29 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v30 = *(_QWORD **)(v29 + 24);
  if ( *(_DWORD *)(v29 + 32) > 0x40u )
    v30 = (_QWORD *)*v30;
  v76 = (int)v30;
  v31 = *a1;
  v32 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 72LL);
  v33 = sub_2E79000(*(__int64 **)(a1[1] + 40));
  v34 = v33;
  if ( v32 == sub_2FE4D20 )
  {
    v35 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v31 + 32LL);
    if ( v35 == sub_2D42F30 )
    {
      v36 = 0;
      v37 = sub_AE2980(v34, 0)[1];
      v38 = 2;
      if ( v37 != 1 )
      {
        v38 = 3;
        if ( v37 != 2 )
        {
          v38 = 4;
          if ( v37 != 4 )
          {
            v38 = 5;
            if ( v37 != 8 )
            {
              v38 = 6;
              if ( v37 != 16 )
              {
                v38 = 7;
                if ( v37 != 32 )
                {
                  v38 = 8;
                  if ( v37 != 64 )
                    v38 = 9 * (v37 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v36 = v34;
      v38 = v35(v31, v34, 0);
    }
  }
  else
  {
    v36 = v33;
    v38 = ((__int64 (__fastcall *)(__int64, __int64))v32)(v31, v33);
  }
  v84 = v38;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v39 = word_4456340[v11 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v91) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v39 = sub_3007130((__int64)&v91, v36);
  }
  v75 = v39;
  v40 = 0;
  v73 = v79;
  if ( v39 )
  {
    do
    {
      v50 = (_QWORD *)a1[1];
      v51 = sub_3400BD0((__int64)v50, v40, (__int64)&v96, v84, 0, 0, v8, 0);
      v53 = v52;
      v54 = v51;
      if ( (_WORD)v89 )
      {
        LOWORD(v41) = word_4456580[(unsigned __int16)v89 - 1];
        v42 = 0;
      }
      else
      {
        v81 = v51;
        v82 = v52;
        v41 = sub_3009970((__int64)&v89, v40, 0, v72, (__int64)v51);
        v54 = v81;
        v53 = v82;
        HIWORD(v2) = HIWORD(v41);
      }
      LOWORD(v2) = v41;
      *((_QWORD *)&v71 + 1) = v53;
      *(_QWORD *)&v71 = v54;
      v87 = v73 | v87 & 0xFFFFFFFF00000000LL;
      v43 = sub_3406EB0(v50, 0x9Eu, (__int64)&v96, v2, v42, v53, __PAIR128__(v87, v78), v71);
      v45 = v44;
      v46 = (unsigned int)(v76 + v40);
      v80 = (_QWORD *)a1[1];
      ++v40;
      *(_QWORD *)&v47 = sub_3400BD0((__int64)v80, v46, (__int64)&v96, v84, 0, 0, v8, 0);
      *((_QWORD *)&v70 + 1) = v45;
      *(_QWORD *)&v70 = v43;
      *(_QWORD *)&v86 = sub_340F900(v80, 0x9Du, (__int64)&v96, v89, v90, v48, v86, v70, v47);
      *((_QWORD *)&v86 + 1) = v49 | *((_QWORD *)&v86 + 1) & 0xFFFFFFFF00000000LL;
    }
    while ( v75 != v40 );
  }
  v64 = v86;
LABEL_55:
  if ( v96.m128i_i64[0] )
    sub_B91220((__int64)&v96, v96.m128i_i64[0]);
  return v64;
}
