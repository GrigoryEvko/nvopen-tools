// Function: sub_3471840
// Address: 0x3471840
//
unsigned __int8 *__fastcall sub_3471840(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4)
{
  unsigned __int8 *result; // rax
  __int64 v8; // rsi
  __int64 *v9; // rdi
  __int64 v10; // r12
  __int16 *v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 (__fastcall *v14)(_QWORD *, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _BOOL4 v19; // r11d
  _BOOL4 v20; // esi
  __int64 v21; // rdx
  unsigned int v22; // r11d
  unsigned int v23; // esi
  __int64 v24; // rbx
  unsigned int v25; // edx
  unsigned int v26; // r12d
  int v27; // eax
  char *v28; // rdx
  unsigned int v29; // edx
  __int128 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned int v33; // r8d
  __int64 v34; // r12
  __int64 v35; // rcx
  unsigned int v36; // edx
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int16 v39; // ax
  __int64 v40; // rdx
  __int64 v41; // rbx
  unsigned int v42; // esi
  unsigned int v43; // edx
  __int64 v44; // rsi
  void *v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdx
  __int128 v48; // rax
  __int64 v49; // r9
  __int64 v50; // rax
  unsigned int v51; // ecx
  __int64 v52; // r8
  __int64 v53; // rsi
  __int64 v54; // r10
  unsigned int v55; // edx
  __int64 v56; // rax
  __int64 v57; // r11
  __int16 v58; // dx
  __int64 v59; // rax
  unsigned int v60; // esi
  unsigned int v61; // edx
  bool v62; // al
  __int64 v63; // rdx
  __int128 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int128 v67; // rax
  __int64 v68; // r9
  unsigned __int8 *v69; // rax
  __int64 v70; // r10
  int v71; // ebx
  __int64 v72; // rdx
  __int128 v73; // rax
  __int64 v74; // r9
  unsigned __int8 *v75; // rax
  __int64 v76; // rdx
  __int128 v77; // rax
  bool v78; // al
  __int128 v79; // [rsp-30h] [rbp-150h]
  __int128 v80; // [rsp-20h] [rbp-140h]
  __int128 v81; // [rsp-20h] [rbp-140h]
  __int128 v82; // [rsp-10h] [rbp-130h]
  int v83; // [rsp-10h] [rbp-130h]
  __int64 v84; // [rsp+8h] [rbp-118h]
  __int64 v85; // [rsp+18h] [rbp-108h]
  __int128 v86; // [rsp+20h] [rbp-100h]
  unsigned int v87; // [rsp+30h] [rbp-F0h]
  char v88; // [rsp+3Fh] [rbp-E1h]
  unsigned int v89; // [rsp+40h] [rbp-E0h]
  __int128 v90; // [rsp+40h] [rbp-E0h]
  void *v91; // [rsp+50h] [rbp-D0h]
  void *v92; // [rsp+50h] [rbp-D0h]
  __int64 v93; // [rsp+50h] [rbp-D0h]
  __int64 v94; // [rsp+50h] [rbp-D0h]
  __int64 v95; // [rsp+50h] [rbp-D0h]
  __int64 v96; // [rsp+58h] [rbp-C8h]
  unsigned int v97; // [rsp+60h] [rbp-C0h]
  __int64 v98; // [rsp+68h] [rbp-B8h]
  unsigned int v99; // [rsp+70h] [rbp-B0h]
  __int64 v100; // [rsp+70h] [rbp-B0h]
  int v101; // [rsp+80h] [rbp-A0h]
  __int128 v102; // [rsp+80h] [rbp-A0h]
  __m128i v103; // [rsp+90h] [rbp-90h]
  __int128 v104; // [rsp+90h] [rbp-90h]
  __m128i v105; // [rsp+A0h] [rbp-80h]
  unsigned __int8 *v106; // [rsp+A0h] [rbp-80h]
  __int64 v107; // [rsp+B0h] [rbp-70h] BYREF
  int v108; // [rsp+B8h] [rbp-68h]
  __int64 v109; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v110; // [rsp+C8h] [rbp-58h]
  void *v111; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v112; // [rsp+D8h] [rbp-48h]

  result = sub_3471020((__int64)a1, a2, (_QWORD *)a3, a4);
  if ( !result )
  {
    v8 = *(_QWORD *)(a2 + 80);
    v107 = v8;
    if ( v8 )
      sub_B96E90((__int64)&v107, v8, 1);
    v9 = *(__int64 **)(a3 + 40);
    v10 = *(_QWORD *)(a3 + 64);
    v108 = *(_DWORD *)(a2 + 72);
    v103 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v101 = *(_DWORD *)(a2 + 24);
    v11 = *(__int16 **)(a2 + 48);
    v105 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
    v12 = *v11;
    v110 = *((_QWORD *)v11 + 1);
    v13 = *a1;
    LOWORD(v109) = v12;
    v14 = *(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int64))(v13 + 528);
    v15 = sub_2E79000(v9);
    v97 = v14(a1, v15, v10, (unsigned int)v109, v110);
    v19 = v101 == 284;
    v20 = v19;
    v98 = v21;
    v99 = *(_DWORD *)(a2 + 28);
    v22 = v19 + 281;
    v23 = v20 + 279;
    if ( (_WORD)v109 == 1 )
    {
      if ( (*((_BYTE *)a1 + v22 + 6914) & 0xFB) == 0 )
      {
LABEL_19:
        v24 = (__int64)sub_3405C90(
                         (_QWORD *)a3,
                         v22,
                         (__int64)&v107,
                         (unsigned int)v109,
                         v110,
                         v99,
                         a4,
                         *(_OWORD *)&v103,
                         *(_OWORD *)&v105);
        v26 = v29;
        if ( (*(_BYTE *)(a2 + 28) & 0x20) != 0 )
        {
LABEL_14:
          result = (unsigned __int8 *)v24;
LABEL_15:
          if ( v107 )
          {
            v106 = result;
            sub_B91220((__int64)&v107, v107);
            return v106;
          }
          return result;
        }
        v88 = 1;
        goto LABEL_28;
      }
      if ( (*((_BYTE *)a1 + v23 + 6914) & 0xFB) != 0 )
      {
LABEL_23:
        if ( (unsigned __int16)(v109 - 17) <= 0xD3u
          && (!a1[(unsigned __int16)v109 + 14] || (*((_BYTE *)a1 + 500 * (unsigned __int16)v109 + 6620) & 0xFB) != 0) )
        {
          goto LABEL_8;
        }
LABEL_24:
        *(_QWORD *)&v30 = sub_33ED040((_QWORD *)a3, 2 * (unsigned int)(v101 != 284) + 2);
        v32 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v107, v97, v98, v31, *(_OWORD *)&v103, *(_OWORD *)&v105, v30);
        v33 = v109;
        v34 = v110;
        v35 = v32;
        v37 = v36;
        v38 = *(_QWORD *)(v32 + 48) + 16LL * v36;
        v39 = *(_WORD *)v38;
        v40 = *(_QWORD *)(v38 + 8);
        v41 = v37;
        LOWORD(v111) = v39;
        v112 = v40;
        if ( v39 )
        {
          v42 = ((unsigned __int16)(v39 - 17) < 0xD4u) + 205;
        }
        else
        {
          v89 = v109;
          v94 = v35;
          v62 = sub_30070B0((__int64)&v111);
          v33 = v89;
          v35 = v94;
          v41 = v37;
          v42 = 205 - (!v62 - 1);
        }
        v24 = sub_340EC60(
                (_QWORD *)a3,
                v42,
                (__int64)&v107,
                v33,
                v34,
                v99,
                v35,
                v41,
                *(_OWORD *)&v103,
                *(_OWORD *)&v105);
        v26 = v43;
        goto LABEL_12;
      }
    }
    else
    {
      if ( !(_WORD)v109 )
      {
        if ( sub_30070B0((__int64)&v109) )
        {
LABEL_8:
          result = sub_3412A00((_QWORD *)a3, a2, 0, v16, v17, v18, a4);
          goto LABEL_15;
        }
        goto LABEL_24;
      }
      v16 = (unsigned __int16)v109;
      if ( !a1[(unsigned __int16)v109 + 14] )
        goto LABEL_23;
      v16 = v22;
      v28 = (char *)a1 + 500 * (unsigned __int16)v109;
      if ( (v28[v22 + 6414] & 0xFB) == 0 )
        goto LABEL_19;
      if ( !a1[(unsigned __int16)v109 + 14] )
        goto LABEL_23;
      v16 = v23;
      if ( (v28[v23 + 6414] & 0xFB) != 0 )
        goto LABEL_23;
    }
    v24 = (__int64)sub_3405C90(
                     (_QWORD *)a3,
                     v23,
                     (__int64)&v107,
                     (unsigned int)v109,
                     v110,
                     v99,
                     a4,
                     *(_OWORD *)&v103,
                     *(_OWORD *)&v105);
    v26 = v25;
LABEL_12:
    v27 = *(_DWORD *)(a2 + 28);
    if ( (v27 & 0x20) != 0 )
    {
LABEL_13:
      if ( (v27 & 0x80u) == 0
        && !(unsigned __int8)sub_33CEB60(a3, v105.m128i_i64[0], v105.m128i_i64[1])
        && !(unsigned __int8)sub_33CEB60(a3, v103.m128i_i64[0], v103.m128i_i64[1]) )
      {
        *(_QWORD *)&v90 = sub_33FE730(a3, (__int64)&v107, (unsigned int)v109, v110, 0, (__m128i)0LL);
        *((_QWORD *)&v90 + 1) = v63;
        *(_QWORD *)&v64 = sub_33ED040((_QWORD *)a3, 1u);
        *((_QWORD *)&v79 + 1) = v26;
        *(_QWORD *)&v79 = v24;
        v65 = 64;
        v95 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v107, v97, v98, *((__int64 *)&v90 + 1), v79, v90, v64);
        if ( v101 != 284 )
          v65 = 32;
        v96 = v66;
        *(_QWORD *)&v67 = sub_3400BD0(a3, v65, (__int64)&v107, 7, 0, 1u, (__m128i)0LL, 0);
        v102 = v67;
        v69 = sub_3406EB0((_QWORD *)a3, 0x9Bu, (__int64)&v107, v97, v98, v68, *(_OWORD *)&v103, v67);
        v70 = v24;
        v71 = v99;
        v83 = v99;
        *((_QWORD *)&v80 + 1) = v26;
        *(_QWORD *)&v80 = v70;
        v100 = v70;
        *(_QWORD *)&v73 = sub_3288B20(a3, (int)&v107, v109, v110, (__int64)v69, v72, *(_OWORD *)&v103, v80, v83);
        v104 = v73;
        v75 = sub_3406EB0((_QWORD *)a3, 0x9Bu, (__int64)&v107, v97, v98, v74, *(_OWORD *)&v105, v102);
        *(_QWORD *)&v77 = sub_3288B20(a3, (int)&v107, v109, v110, (__int64)v75, v76, *(_OWORD *)&v105, v104, v71);
        *((_QWORD *)&v81 + 1) = v26;
        *(_QWORD *)&v81 = v100;
        v24 = sub_3288B20(a3, (int)&v107, v109, v110, v95, v96, v77, v81, v71);
      }
      goto LABEL_14;
    }
    v88 = 0;
LABEL_28:
    v44 = v105.m128i_i64[0];
    if ( !(unsigned __int8)sub_33CE830((_QWORD **)a3, v105.m128i_i64[0], v105.m128i_i64[1], 0, 0)
      || (v44 = v103.m128i_i64[0],
          !(unsigned __int8)sub_33CE830((_QWORD **)a3, v103.m128i_i64[0], v103.m128i_i64[1], 0, 0)) )
    {
      v91 = sub_300AC80((unsigned __int16 *)&v109, v44);
      v45 = sub_C33340();
      v46 = (__int64)v91;
      if ( v91 == v45 )
      {
        v92 = v45;
        sub_C3C500(&v111, (__int64)v45);
      }
      else
      {
        v92 = v45;
        sub_C373C0(&v111, v46);
      }
      if ( v111 == v92 )
        sub_C3D480((__int64)&v111, 0, 0, 0);
      else
        sub_C36070((__int64)&v111, 0, 0, 0);
      v93 = sub_AC8EA0(*(__int64 **)(a3 + 64), (__int64 *)&v111);
      sub_91D830(&v111);
      *(_QWORD *)&v86 = sub_33FE020(a3, v93, (__int64)&v107, (unsigned int)v109, v110, 0, a4);
      *((_QWORD *)&v86 + 1) = v47;
      *(_QWORD *)&v48 = sub_33ED040((_QWORD *)a3, 8u);
      v50 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v107, v97, v98, v49, *(_OWORD *)&v103, *(_OWORD *)&v105, v48);
      v51 = v109;
      v52 = v110;
      v53 = v50;
      v54 = v50;
      v56 = *(_QWORD *)(v50 + 48) + 16LL * v55;
      v57 = v55;
      v58 = *(_WORD *)v56;
      v59 = *(_QWORD *)(v56 + 8);
      LOWORD(v111) = v58;
      v112 = v59;
      if ( v58 )
      {
        v60 = ((unsigned __int16)(v58 - 17) < 0xD4u) + 205;
      }
      else
      {
        v84 = v110;
        v87 = v109;
        v85 = v57;
        v78 = sub_30070B0((__int64)&v111);
        v52 = v84;
        v51 = v87;
        v54 = v53;
        v57 = v85;
        v60 = 205 - (!v78 - 1);
      }
      *((_QWORD *)&v82 + 1) = v26;
      *(_QWORD *)&v82 = v24;
      v24 = sub_340EC60((_QWORD *)a3, v60, (__int64)&v107, v51, v52, v99, v54, v57, v86, v82);
      v26 = v61;
    }
    if ( v88 )
      goto LABEL_14;
    v27 = *(_DWORD *)(a2 + 28);
    goto LABEL_13;
  }
  return result;
}
