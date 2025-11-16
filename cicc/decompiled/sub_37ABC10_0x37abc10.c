// Function: sub_37ABC10
// Address: 0x37abc10
//
unsigned __int8 *__fastcall sub_37ABC10(__int64 *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rsi
  int v4; // eax
  const __m128i *v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r14
  __int64 v13; // r12
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // r15
  unsigned __int16 *v16; // rbx
  unsigned int v17; // esi
  unsigned __int16 *v18; // rax
  __int64 v19; // rsi
  unsigned __int64 v20; // rbx
  unsigned __int16 *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int16 *v24; // rsi
  int v25; // eax
  unsigned int v26; // eax
  __int64 v27; // rdx
  unsigned int v28; // r9d
  __int64 v29; // rax
  __int64 v30; // r9
  unsigned int v31; // ecx
  __int64 v32; // rdx
  __int16 v33; // ax
  unsigned __int32 v34; // eax
  unsigned __int32 i; // ecx
  int v36; // edx
  __int64 *v37; // rax
  unsigned __int16 v38; // ax
  __int64 v39; // r8
  unsigned int v40; // r9d
  _QWORD *v41; // rdi
  __int128 v42; // rax
  unsigned __int32 v43; // r9d
  __int64 v44; // rbx
  unsigned __int32 v45; // r12d
  __int128 v46; // rax
  __int64 v47; // r9
  unsigned int v48; // edx
  unsigned __int8 *v49; // r14
  bool v51; // al
  __int64 v52; // r12
  __int64 v53; // rsi
  __int128 v54; // rax
  __int64 v55; // r9
  __int64 v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // rax
  __int64 *v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdx
  __m128i v62; // rax
  __int64 v63; // r12
  int v64; // r10d
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // rax
  __int64 (*v67)(void); // rdx
  unsigned __int16 v68; // ax
  unsigned __int8 *v69; // rax
  __m128i v70; // xmm1
  __int64 v71; // rdx
  _QWORD *v72; // rdi
  __m128i v73; // xmm2
  __int128 v74; // rax
  __int64 v75; // rdx
  __int128 v76; // [rsp-30h] [rbp-180h]
  __int128 v77; // [rsp-30h] [rbp-180h]
  __int128 v78; // [rsp-10h] [rbp-160h]
  __int64 *v79; // [rsp+0h] [rbp-150h]
  unsigned int v80; // [rsp+8h] [rbp-148h]
  __m128i v81; // [rsp+10h] [rbp-140h] BYREF
  __int64 v82; // [rsp+20h] [rbp-130h]
  unsigned int v83; // [rsp+28h] [rbp-128h]
  unsigned int v84; // [rsp+2Ch] [rbp-124h]
  __int128 v85; // [rsp+30h] [rbp-120h]
  int v86; // [rsp+40h] [rbp-110h]
  unsigned int v87; // [rsp+44h] [rbp-10Ch]
  _QWORD *v88; // [rsp+48h] [rbp-108h]
  __m128i v89; // [rsp+50h] [rbp-100h] BYREF
  unsigned __int64 v90; // [rsp+60h] [rbp-F0h]
  __int64 *v91; // [rsp+68h] [rbp-E8h]
  __int64 v92; // [rsp+78h] [rbp-D8h]
  __int64 v93; // [rsp+80h] [rbp-D0h]
  __int64 v94; // [rsp+88h] [rbp-C8h]
  __int64 v95; // [rsp+90h] [rbp-C0h]
  __int64 v96; // [rsp+98h] [rbp-B8h]
  __int64 v97; // [rsp+A0h] [rbp-B0h]
  __int64 v98; // [rsp+A8h] [rbp-A8h]
  __int64 v99; // [rsp+B0h] [rbp-A0h] BYREF
  int v100; // [rsp+B8h] [rbp-98h]
  unsigned __int16 v101; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v102; // [rsp+C8h] [rbp-88h]
  unsigned int v103; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v104; // [rsp+D8h] [rbp-78h]
  __m128i v105; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v106; // [rsp+F0h] [rbp-60h]
  unsigned __int64 v107; // [rsp+F8h] [rbp-58h]
  __m128i v108; // [rsp+100h] [rbp-50h]
  unsigned __int8 *v109; // [rsp+110h] [rbp-40h]
  __int64 v110; // [rsp+118h] [rbp-38h]

  v2 = a2;
  v3 = *(_QWORD *)(a2 + 80);
  v91 = a1;
  v99 = v3;
  if ( v3 )
  {
    v89.m128i_i64[0] = v2;
    sub_B96E90((__int64)&v99, v3, 1);
    v2 = v89.m128i_i64[0];
  }
  v4 = *(_DWORD *)(v2 + 72);
  v88 = (_QWORD *)v2;
  v100 = v4;
  v5 = *(const __m128i **)(v2 + 40);
  v6 = _mm_loadu_si128(v5);
  v7 = v5[3].m128i_i64[0];
  v8 = v5[3].m128i_u32[0];
  v89.m128i_i64[0] = v5[2].m128i_i64[1];
  v81 = v6;
  v9 = sub_379AB60((__int64)v91, v89.m128i_u64[0], v7);
  v10 = v88;
  v11 = v89.m128i_i64[0];
  v12 = v9;
  v13 = v9;
  v15 = v14;
  v18 = (unsigned __int16 *)v88[6];
  v16 = (unsigned __int16 *)(*(_QWORD *)(v89.m128i_i64[0] + 48) + 16 * v8);
  v82 = *((_QWORD *)v18 + 1);
  v17 = *v18;
  LODWORD(v18) = *v16;
  v80 = v17;
  v19 = *((_QWORD *)v16 + 1);
  v20 = (unsigned int)v14;
  v101 = (unsigned __int16)v18;
  v21 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * (unsigned int)v14);
  v102 = v19;
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  LOWORD(v103) = v22;
  v104 = v23;
  if ( (_WORD)v18 )
  {
    v88 = 0;
    LOWORD(v18) = word_4456580[(int)v18 - 1];
  }
  else
  {
    v89.m128i_i64[0] = (__int64)v88;
    v18 = (unsigned __int16 *)sub_3009970((__int64)&v101, v22, v23, (__int64)v88, v11);
    v10 = v88;
    v90 = (unsigned __int64)v18;
    v88 = v57;
  }
  v24 = (unsigned __int16 *)v90;
  LOWORD(v24) = (_WORD)v18;
  v25 = *((_DWORD *)v10 + 7);
  v84 = *((_DWORD *)v10 + 6);
  v90 = (unsigned __int64)v24;
  v86 = v25;
  v26 = sub_33CB000(v84);
  *(_QWORD *)&v85 = sub_3401F50(v91[1], v26, (__int64)&v99, (unsigned int)v24, (__int64)v88, v86, v6);
  *((_QWORD *)&v85 + 1) = v27;
  if ( v101 )
  {
    v28 = word_4456340[v101 - 1];
  }
  else
  {
    v96 = sub_3007240((__int64)&v101);
    v28 = v96;
  }
  if ( (_WORD)v103 )
  {
    v89.m128i_i32[0] = word_4456340[(unsigned __int16)v103 - 1];
  }
  else
  {
    v87 = v28;
    v56 = sub_3007240((__int64)&v103);
    v28 = v87;
    v95 = v56;
    v89.m128i_i32[0] = v56;
  }
  v87 = v28;
  v29 = sub_33CB7C0(v84);
  v30 = v87;
  v92 = v29;
  if ( !BYTE4(v29) )
  {
    v33 = v103;
    if ( (_WORD)v103 )
      goto LABEL_13;
    goto LABEL_34;
  }
  v31 = 1;
  v32 = *v91;
  v33 = v103;
  if ( (_WORD)v103 == 1 )
    goto LABEL_11;
  if ( !(_WORD)v103 )
  {
LABEL_34:
    v51 = sub_3007100((__int64)&v103);
    v30 = v87;
    if ( v51 )
      goto LABEL_14;
    goto LABEL_35;
  }
  v31 = (unsigned __int16)v103;
  if ( !*(_QWORD *)(v32 + 8LL * (unsigned __int16)v103 + 112) )
    goto LABEL_13;
LABEL_11:
  if ( (unsigned int)v92 <= 0x1F3 && (*(_BYTE *)((unsigned int)v92 + 500LL * v31 + v32 + 6414) & 0xFB) != 0 )
  {
LABEL_13:
    if ( (unsigned __int16)(v33 - 176) <= 0x34u )
    {
LABEL_14:
      if ( (_DWORD)v30 )
      {
        if ( v89.m128i_i32[0] )
        {
          v34 = v89.m128i_i32[0];
          for ( i = (unsigned int)v30 % v89.m128i_i32[0]; i; i = v36 )
          {
            v36 = v34 % i;
            v34 = i;
          }
          v87 = v34;
        }
        else
        {
          v87 = v30;
        }
      }
      else
      {
        v87 = v89.m128i_i32[0];
      }
      v83 = v30;
      v37 = *(__int64 **)(v91[1] + 64);
      v105.m128i_i8[4] = 1;
      v79 = v37;
      v105.m128i_i32[0] = v87;
      v38 = sub_2D43AD0(v90, v87);
      v39 = 0;
      v40 = v83;
      if ( !v38 )
      {
        v38 = sub_3009450(v79, (unsigned int)v90, (__int64)v88, v105.m128i_i64[0], 0, v83);
        v40 = v83;
        v39 = v75;
      }
      LODWORD(v90) = v40;
      v41 = (_QWORD *)v91[1];
      if ( *(_DWORD *)(v85 + 24) == 51 )
      {
        v105.m128i_i64[0] = 0;
        v105.m128i_i32[2] = 0;
        v88 = &v105;
        *(_QWORD *)&v74 = sub_33F17F0(v41, 51, (__int64)&v105, v38, v39);
        v43 = v90;
        v85 = v74;
        if ( v105.m128i_i64[0] )
        {
          sub_B91220((__int64)v88, v105.m128i_i64[0]);
          v43 = v90;
        }
      }
      else
      {
        *(_QWORD *)&v42 = sub_33FAF80((__int64)v41, 168, (__int64)&v99, v38, v39, v40, v6);
        v43 = v90;
        v85 = v42;
      }
      if ( v43 >= v89.m128i_i32[0] )
        goto LABEL_29;
      v90 = v20;
      v44 = v13;
      v45 = v43;
      while ( 1 )
      {
        v88 = (_QWORD *)v91[1];
        *(_QWORD *)&v46 = sub_3400EE0((__int64)v88, v45, (__int64)&v99, 0, v6);
        v15 = v15 & 0xFFFFFFFF00000000LL | v90;
        *((_QWORD *)&v76 + 1) = v15;
        *(_QWORD *)&v76 = v44;
        v45 += v87;
        v44 = sub_340F900(v88, 0xA0u, (__int64)&v99, v103, v104, v47, v76, v85, v46);
        if ( v45 >= v89.m128i_i32[0] )
          break;
        v90 = v48;
      }
      goto LABEL_28;
    }
LABEL_35:
    if ( (unsigned int)v30 >= v89.m128i_i32[0] )
      goto LABEL_29;
    v90 = v20;
    v44 = v13;
    v52 = v30;
    while ( 1 )
    {
      v53 = v52++;
      v88 = (_QWORD *)v91[1];
      *(_QWORD *)&v54 = sub_3400EE0((__int64)v88, v53, (__int64)&v99, 0, v6);
      v15 = v15 & 0xFFFFFFFF00000000LL | v90;
      *((_QWORD *)&v77 + 1) = v15;
      *(_QWORD *)&v77 = v44;
      v44 = sub_340F900(v88, 0x9Du, (__int64)&v99, v103, v104, v55, v77, v85, v54);
      if ( v89.m128i_i32[0] <= (unsigned int)v52 )
        break;
      v90 = v48;
    }
LABEL_28:
    v13 = v44;
    v20 = v48;
LABEL_29:
    *((_QWORD *)&v78 + 1) = v20 | v15 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v78 = v13;
    v49 = sub_3405C90((_QWORD *)v91[1], v84, (__int64)&v99, v80, v82, v86, v6, *(_OWORD *)&v81, v78);
    goto LABEL_30;
  }
  LODWORD(v90) = v92;
  v58 = v91[1];
  BYTE4(v98) = (unsigned __int16)(v103 - 176) <= 0x34u;
  v59 = *(__int64 **)(v58 + 64);
  LODWORD(v98) = word_4456340[v31 - 1];
  v93 = v98;
  v60 = sub_327FD70(v59, 2u, 0, v98);
  v62.m128i_i64[0] = (__int64)sub_34015B0(v91[1], (__int64)&v99, v60, v61, 0, 0, v6);
  v63 = v91[1];
  v64 = v92;
  v89 = v62;
  if ( v101 )
  {
    LOBYTE(v65) = (unsigned __int16)(v101 - 176) <= 0x34u;
    LODWORD(v66) = word_4456340[v101 - 1];
  }
  else
  {
    v66 = sub_3007240((__int64)&v101);
    v64 = v90;
    v97 = v66;
    v65 = HIDWORD(v66);
  }
  BYTE4(v97) = v65;
  LODWORD(v97) = v66;
  v94 = v97;
  v67 = *(__int64 (**)(void))(*(_QWORD *)*v91 + 80LL);
  v68 = 7;
  if ( v67 != sub_2FE2E20 )
  {
    LODWORD(v90) = v64;
    v68 = v67();
    v64 = v90;
  }
  LODWORD(v90) = v64;
  v69 = sub_3401C20(v63, (__int64)&v99, v68, 0, v94, v6);
  v106 = v12;
  v109 = v69;
  v70 = _mm_load_si128(&v81);
  v110 = v71;
  v72 = (_QWORD *)v91[1];
  v73 = _mm_load_si128(&v89);
  v107 = v15;
  v105 = v70;
  v108 = v73;
  v49 = sub_33FBA10(v72, (unsigned int)v90, (__int64)&v99, v80, v82, v86, (__int64)&v105, 4);
LABEL_30:
  if ( v99 )
    sub_B91220((__int64)&v99, v99);
  return v49;
}
