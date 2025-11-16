// Function: sub_1F823C0
// Address: 0x1f823c0
//
__int64 *__fastcall sub_1F823C0(__int64 a1, __int64 a2, __m128i a3, __m128 a4, __m128i a5)
{
  __int64 *v5; // r13
  __int64 **v7; // r9
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // r9
  int v14; // r12d
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  const __m128i *v18; // r9
  const __m128i *v19; // r15
  const __m128i *v20; // rbx
  __int64 v21; // r8
  __int16 v22; // ax
  bool v23; // al
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // rdx
  _QWORD *v29; // rdx
  __int64 *v30; // rax
  char v31; // dl
  __int64 *v32; // rsi
  __int64 *v33; // rcx
  __int64 v34; // rax
  const __m128i *v35; // rbx
  _BYTE **v36; // r9
  unsigned int v37; // esi
  __int64 v38; // rdx
  unsigned int v39; // ecx
  const __m128i *v40; // r12
  __int64 v41; // r8
  __int64 v42; // rax
  __int64 *v43; // rdx
  __int64 v44; // rdx
  int v45; // r9d
  __int64 v46; // r12
  int v47; // r15d
  _DWORD *v48; // rax
  __int64 v49; // rdx
  unsigned int v50; // r13d
  int v51; // eax
  __int64 v52; // rbx
  __int64 *v53; // r12
  __int64 *v54; // rbx
  __int64 v55; // rdx
  unsigned __int64 v56; // rcx
  __int64 v57; // rax
  unsigned int v58; // ebx
  unsigned __int64 v59; // r13
  __int64 v60; // rsi
  __int64 *v61; // r15
  const __m128i *v62; // r13
  const __m128i *v63; // rbx
  int v64; // r8d
  int v65; // r9d
  __int64 v66; // rax
  __int128 v67; // rcx
  __int64 *v68; // r13
  __int64 v69; // rsi
  __int128 v70; // [rsp-10h] [rbp-490h]
  __int64 v71; // [rsp+8h] [rbp-478h]
  __int64 v72; // [rsp+10h] [rbp-470h]
  __int64 v73; // [rsp+20h] [rbp-460h]
  __int64 v74; // [rsp+40h] [rbp-440h]
  __int128 v77; // [rsp+50h] [rbp-430h]
  char v78; // [rsp+6Ah] [rbp-416h] BYREF
  char v79; // [rsp+6Bh] [rbp-415h] BYREF
  unsigned int v80; // [rsp+6Ch] [rbp-414h] BYREF
  __int64 v81; // [rsp+70h] [rbp-410h] BYREF
  int v82; // [rsp+78h] [rbp-408h]
  _BYTE *v83; // [rsp+80h] [rbp-400h] BYREF
  __int64 v84; // [rsp+88h] [rbp-3F8h]
  _BYTE v85[32]; // [rsp+90h] [rbp-3F0h] BYREF
  __int64 *v86; // [rsp+B0h] [rbp-3D0h] BYREF
  char *v87; // [rsp+B8h] [rbp-3C8h]
  char *v88; // [rsp+C0h] [rbp-3C0h]
  const __m128i **v89; // [rsp+C8h] [rbp-3B8h]
  _BYTE **v90; // [rsp+D0h] [rbp-3B0h]
  _BYTE **v91; // [rsp+D8h] [rbp-3A8h]
  unsigned int *v92; // [rsp+E0h] [rbp-3A0h]
  __int64 *v93; // [rsp+E8h] [rbp-398h]
  _QWORD *v94; // [rsp+F0h] [rbp-390h] BYREF
  __int64 v95; // [rsp+F8h] [rbp-388h]
  _QWORD v96[8]; // [rsp+100h] [rbp-380h] BYREF
  const __m128i *v97; // [rsp+140h] [rbp-340h] BYREF
  __int64 v98; // [rsp+148h] [rbp-338h]
  _BYTE v99[128]; // [rsp+150h] [rbp-330h] BYREF
  _BYTE *v100; // [rsp+1D0h] [rbp-2B0h] BYREF
  __int64 v101; // [rsp+1D8h] [rbp-2A8h]
  _BYTE v102[128]; // [rsp+1E0h] [rbp-2A0h] BYREF
  _BYTE *v103; // [rsp+260h] [rbp-220h] BYREF
  __int64 v104; // [rsp+268h] [rbp-218h]
  _BYTE v105[128]; // [rsp+270h] [rbp-210h] BYREF
  __int64 v106; // [rsp+2F0h] [rbp-190h] BYREF
  __int64 *v107; // [rsp+2F8h] [rbp-188h]
  __int64 *v108; // [rsp+300h] [rbp-180h]
  __int64 v109; // [rsp+308h] [rbp-178h]
  int v110; // [rsp+310h] [rbp-170h]
  _BYTE v111[136]; // [rsp+318h] [rbp-168h] BYREF
  __int64 v112; // [rsp+3A0h] [rbp-E0h] BYREF
  _BYTE *v113; // [rsp+3A8h] [rbp-D8h]
  _BYTE *v114; // [rsp+3B0h] [rbp-D0h]
  __int64 v115; // [rsp+3B8h] [rbp-C8h]
  int v116; // [rsp+3C0h] [rbp-C0h]
  _BYTE v117[184]; // [rsp+3C8h] [rbp-B8h] BYREF

  if ( *(_DWORD *)(a2 + 56) == 2 )
  {
    v7 = *(__int64 ***)(a2 + 32);
    v5 = *v7;
    v9 = sub_1F6C710((unsigned int *)(*v7)[4], *((_DWORD *)*v7 + 14));
    v11 = *(_QWORD *)(v10 + 40);
    if ( v11 == v9 && *(_DWORD *)(v10 + 48) == v8 )
      return v5;
    if ( v5 == (__int64 *)sub_1F6C710(*(unsigned int **)(v11 + 32), *(_DWORD *)(v11 + 56))
      && *(_DWORD *)(v13 + 8) == v12 )
    {
      return *(__int64 **)(v13 + 40);
    }
  }
  if ( !*(_DWORD *)(a1 + 20) )
    return 0;
  v14 = 0;
  v106 = 0;
  v15 = v96;
  v97 = (const __m128i *)v99;
  v98 = 0x800000000LL;
  v107 = (__int64 *)v111;
  v108 = (__int64 *)v111;
  v94 = v96;
  v109 = 16;
  v110 = 0;
  v78 = 0;
  v96[0] = a2;
  v95 = 0x800000001LL;
  v16 = 0;
  while ( 1 )
  {
    v17 = v15[v16];
    v18 = *(const __m128i **)(v17 + 32);
    v19 = v18;
    v20 = (const __m128i *)((char *)v18 + 40 * *(unsigned int *)(v17 + 56));
    if ( v18 != v20 )
    {
      while ( 1 )
      {
        v21 = v19->m128i_i64[0];
        v22 = *(_WORD *)(v19->m128i_i64[0] + 24);
        if ( v22 == 1 )
          goto LABEL_24;
        if ( v22 == 2 )
        {
          v23 = sub_1D18C00(v19->m128i_i64[0], 1, v19->m128i_i32[2]);
          v21 = v19->m128i_i64[0];
          if ( v23 )
            break;
        }
LABEL_22:
        v30 = v107;
        if ( v108 == v107 )
        {
          v32 = &v107[HIDWORD(v109)];
          if ( v107 != v32 )
          {
            v33 = 0;
            while ( *v30 != v21 )
            {
              if ( *v30 == -2 )
                v33 = v30;
              if ( v32 == ++v30 )
              {
                if ( !v33 )
                  goto LABEL_46;
                *v33 = v21;
                --v110;
                ++v106;
                goto LABEL_36;
              }
            }
            goto LABEL_24;
          }
LABEL_46:
          if ( HIDWORD(v109) < (unsigned int)v109 )
          {
            ++HIDWORD(v109);
            *v32 = v21;
            v34 = (unsigned int)v98;
            ++v106;
            if ( (unsigned int)v98 >= HIDWORD(v98) )
            {
LABEL_48:
              sub_16CD150((__int64)&v97, v99, 0, 16, v21, (int)v18);
              v34 = (unsigned int)v98;
            }
LABEL_37:
            a3 = _mm_loadu_si128(v19);
            v97[v34] = a3;
            LODWORD(v98) = v98 + 1;
            goto LABEL_25;
          }
        }
        sub_16CCBA0((__int64)&v106, v21);
        if ( v31 )
        {
LABEL_36:
          v34 = (unsigned int)v98;
          if ( (unsigned int)v98 >= HIDWORD(v98) )
            goto LABEL_48;
          goto LABEL_37;
        }
LABEL_24:
        v78 = 1;
LABEL_25:
        v19 = (const __m128i *)((char *)v19 + 40);
        if ( v20 == v19 )
          goto LABEL_26;
      }
      v24 = v94;
      v25 = 8LL * (unsigned int)v95;
      v26 = &v94[(unsigned __int64)v25 / 8];
      v27 = v25 >> 3;
      v28 = v25 >> 5;
      if ( v28 )
      {
        v29 = &v94[4 * v28];
        while ( v21 != *v24 )
        {
          if ( v21 == v24[1] )
          {
            ++v24;
            break;
          }
          if ( v21 == v24[2] )
          {
            v24 += 2;
            break;
          }
          if ( v21 == v24[3] )
          {
            v24 += 3;
            break;
          }
          v24 += 4;
          if ( v24 == v29 )
          {
            v27 = v26 - v24;
            goto LABEL_39;
          }
        }
LABEL_21:
        if ( v26 == v24 )
          goto LABEL_43;
        goto LABEL_22;
      }
LABEL_39:
      if ( v27 != 2 )
      {
        if ( v27 != 3 )
        {
          if ( v27 != 1 )
            goto LABEL_43;
          goto LABEL_42;
        }
        if ( v21 == *v24 )
          goto LABEL_21;
        ++v24;
      }
      if ( v21 == *v24 )
        goto LABEL_21;
      ++v24;
LABEL_42:
      if ( v21 != *v24 )
      {
LABEL_43:
        if ( (unsigned int)v95 >= HIDWORD(v95) )
        {
          v73 = v19->m128i_i64[0];
          sub_16CD150((__int64)&v94, v96, 0, 8, v21, (int)v18);
          v21 = v73;
          v26 = &v94[(unsigned int)v95];
        }
        *v26 = v21;
        LODWORD(v95) = v95 + 1;
        sub_1F81BC0(a1, v19->m128i_i64[0]);
        v78 = 1;
        goto LABEL_25;
      }
      goto LABEL_21;
    }
LABEL_26:
    v16 = (unsigned int)(v14 + 1);
    v14 = v16;
    if ( (unsigned int)v16 >= (unsigned int)v95 )
      break;
    v15 = v94;
  }
  v35 = v97;
  v112 = 0;
  v36 = &v100;
  v100 = v102;
  v101 = 0x800000000LL;
  v84 = 0x800000000LL;
  v113 = v117;
  v114 = v117;
  v83 = v85;
  v79 = 0;
  v115 = 16;
  v116 = 0;
  v80 = 0;
  if ( v97 == &v97[(unsigned int)v98] )
  {
    v86 = &v106;
    v87 = &v78;
    v88 = &v79;
    v90 = &v100;
    v89 = &v97;
    v92 = &v80;
    v91 = &v83;
    v93 = &v112;
  }
  else
  {
    v37 = 8;
    v38 = 0;
    v39 = 0;
    v40 = &v97[(unsigned int)v98];
    while ( 1 )
    {
      v41 = v39;
      v80 = v39 + 1;
      v42 = v35->m128i_i64[0];
      if ( (unsigned int)v38 >= v37 )
      {
        v71 = v35->m128i_i64[0];
        v72 = v39;
        sub_16CD150((__int64)&v100, v102, 0, 16, v39, (int)v36);
        v38 = (unsigned int)v101;
        v42 = v71;
        v41 = v72;
      }
      v43 = (__int64 *)&v100[16 * v38];
      *v43 = v42;
      v43[1] = v41;
      v44 = (unsigned int)v84;
      LODWORD(v101) = v101 + 1;
      if ( (unsigned int)v84 >= HIDWORD(v84) )
      {
        sub_16CD150((__int64)&v83, v85, 0, 4, v41, (int)v36);
        v44 = (unsigned int)v84;
      }
      ++v35;
      *(_DWORD *)&v83[4 * v44] = 1;
      LODWORD(v84) = v84 + 1;
      if ( v40 == v35 )
        break;
      v39 = v80;
      v38 = (unsigned int)v101;
      v37 = HIDWORD(v101);
    }
    v45 = v101;
    v86 = &v106;
    v87 = &v78;
    v88 = &v79;
    v90 = &v100;
    v89 = &v97;
    v92 = &v80;
    v91 = &v83;
    v93 = &v112;
    if ( (_DWORD)v101 )
    {
      v46 = 0;
      while ( 1 )
      {
        v47 = v46;
        if ( v80 <= 1 )
          goto LABEL_88;
        v48 = &v100[16 * v46];
        v49 = *(_QWORD *)v48;
        v50 = v48[2];
        v51 = *(unsigned __int16 *)(*(_QWORD *)v48 + 24LL);
        if ( (_WORD)v51 == 2 )
        {
          v52 = *(_QWORD *)(v49 + 32);
          if ( v52 != v52 + 40LL * *(unsigned int *)(v49 + 56) )
          {
            v74 = v46;
            v53 = *(__int64 **)(v49 + 32);
            v54 = (__int64 *)(v52 + 40LL * *(unsigned int *)(v49 + 56));
            do
            {
              v55 = *v53;
              v53 += 5;
              sub_1F6EC10((__int64 *)&v86, v47, v55, v50, v41, v45);
            }
            while ( v54 != v53 );
            v46 = v74;
          }
          goto LABEL_68;
        }
        if ( (__int16)v51 > 2 )
          break;
        if ( (_WORD)v51 != 1 )
          goto LABEL_81;
        ++v80;
LABEL_68:
        v41 = (unsigned int)--*(_DWORD *)&v83[4 * v50];
        if ( !(_DWORD)v41 )
          --v80;
        if ( (unsigned int)v101 <= (unsigned int)++v46 || (_DWORD)v46 == 1024 )
          goto LABEL_88;
      }
      if ( (unsigned __int16)(v51 - 46) > 1u )
      {
        v56 = (unsigned int)(v51 - 185);
        if ( (unsigned __int16)(v51 - 185) > 0x35u )
        {
LABEL_81:
          if ( (unsigned __int16)(v51 - 44) <= 1u )
          {
            if ( (*(_BYTE *)(v49 + 26) & 2) == 0 )
              goto LABEL_68;
          }
          else if ( (__int16)v51 <= 658 )
          {
            goto LABEL_68;
          }
        }
        else
        {
          v57 = 0x3FFFFD00000003LL;
          if ( !_bittest64(&v57, v56) )
            goto LABEL_68;
        }
      }
      sub_1F6EC10((__int64 *)&v86, v46, **(_QWORD **)(v49 + 32), v50, v41, v45);
      goto LABEL_68;
    }
  }
LABEL_88:
  if ( v78 )
  {
    v58 = v98;
    if ( (_DWORD)v98 )
    {
      if ( v79 )
      {
        v103 = v105;
        v62 = v97;
        v104 = 0x800000000LL;
        v63 = &v97[(unsigned int)v98];
        do
        {
          if ( !sub_1F816C0((__int64)&v112, v62->m128i_i64[0]) )
          {
            v66 = (unsigned int)v104;
            if ( (unsigned int)v104 >= HIDWORD(v104) )
            {
              sub_16CD150((__int64)&v103, v105, 0, 16, v64, v65);
              v66 = (unsigned int)v104;
            }
            a4 = (__m128)_mm_loadu_si128(v62);
            *(__m128 *)&v103[16 * v66] = a4;
            LODWORD(v104) = v104 + 1;
          }
          ++v62;
        }
        while ( v63 != v62 );
        *(_QWORD *)&v67 = v103;
        *((_QWORD *)&v67 + 1) = (unsigned int)v104;
        v68 = *(__int64 **)a1;
        v69 = *(_QWORD *)(a2 + 72);
        v81 = v69;
        if ( v69 )
        {
          *(_QWORD *)&v77 = v103;
          *((_QWORD *)&v77 + 1) = (unsigned int)v104;
          sub_1623A60((__int64)&v81, v69, 2);
          v67 = v77;
        }
        v82 = *(_DWORD *)(a2 + 64);
        v5 = sub_1D359D0(v68, 2, (__int64)&v81, 1, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128_u64, a5, v67);
        if ( v81 )
          sub_161E7C0((__int64)&v81, v81);
        if ( v103 != v105 )
          _libc_free((unsigned __int64)v103);
      }
      else
      {
        v59 = (unsigned __int64)v97;
        v60 = *(_QWORD *)(a2 + 72);
        v61 = *(__int64 **)a1;
        v103 = (_BYTE *)v60;
        if ( v60 )
          sub_1623A60((__int64)&v103, v60, 2);
        *((_QWORD *)&v70 + 1) = v58;
        *(_QWORD *)&v70 = v59;
        LODWORD(v104) = *(_DWORD *)(a2 + 64);
        v5 = sub_1D359D0(v61, 2, (__int64)&v103, 1, 0, 0, *(double *)a3.m128i_i64, *(double *)a4.m128_u64, a5, v70);
        if ( v103 )
          sub_161E7C0((__int64)&v103, (__int64)v103);
      }
    }
    else
    {
      v5 = (__int64 *)(*(_QWORD *)a1 + 88LL);
    }
  }
  else
  {
    v5 = 0;
  }
  if ( v114 != v113 )
    _libc_free((unsigned __int64)v114);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  if ( v108 != v107 )
    _libc_free((unsigned __int64)v108);
  if ( v97 != (const __m128i *)v99 )
    _libc_free((unsigned __int64)v97);
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
  return v5;
}
