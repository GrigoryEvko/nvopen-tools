// Function: sub_27AAC70
// Address: 0x27aac70
//
__int64 __fastcall sub_27AAC70(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  _BYTE *v3; // rsi
  char *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  const __m128i *v10; // rcx
  const __m128i *v11; // rdx
  __int64 v12; // r11
  unsigned __int64 v13; // rbx
  __m128i *v14; // rax
  __int64 v15; // rcx
  const __m128i *v16; // rax
  const __m128i *v17; // rcx
  __int64 v18; // r11
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  __m128i *v22; // rdx
  __m128i *v23; // rax
  __int64 v24; // r13
  unsigned __int64 v25; // rcx
  unsigned int v26; // esi
  __int64 v27; // rbx
  __int64 v28; // r8
  int v29; // r11d
  _QWORD *v30; // rdx
  unsigned int v31; // edi
  _QWORD *v32; // rax
  __int64 v33; // rcx
  _DWORD *v34; // rax
  __int64 v35; // r11
  int v36; // r14d
  __int64 v37; // r12
  __int64 v38; // r8
  unsigned int v39; // edi
  __int64 *v40; // rax
  __int64 v41; // rcx
  unsigned int v42; // esi
  __int64 v43; // rbx
  int v44; // esi
  int v45; // esi
  __int64 v46; // r8
  unsigned int v47; // ecx
  int v48; // eax
  __int64 *v49; // rdx
  __int64 v50; // rdi
  int v51; // eax
  int v52; // ecx
  int v53; // eax
  int v54; // ecx
  int v55; // ecx
  __int64 v56; // rdi
  __int64 *v57; // r8
  unsigned int v58; // r15d
  int v59; // r9d
  __int64 v60; // rsi
  unsigned __int64 v61; // rax
  char v62; // si
  __int64 v63; // r11
  int v64; // ebx
  unsigned int v65; // r12d
  __int64 v66; // r13
  unsigned __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  int v71; // eax
  int v72; // edi
  __int64 v73; // rsi
  unsigned int v74; // eax
  __int64 v75; // r8
  int v76; // r10d
  _QWORD *v77; // r9
  int v78; // eax
  int v79; // eax
  __int64 v80; // rdi
  _QWORD *v81; // r8
  unsigned int v82; // r12d
  int v83; // r9d
  __int64 v84; // rsi
  int v85; // r15d
  __int64 *v86; // r9
  __int64 v89; // [rsp+10h] [rbp-240h]
  int v90; // [rsp+10h] [rbp-240h]
  __int64 v91; // [rsp+10h] [rbp-240h]
  __int64 v92; // [rsp+18h] [rbp-238h]
  __int64 v94; // [rsp+28h] [rbp-228h]
  int v95; // [rsp+28h] [rbp-228h]
  __int64 v96; // [rsp+28h] [rbp-228h]
  __int64 v97; // [rsp+28h] [rbp-228h]
  __int64 v98; // [rsp+28h] [rbp-228h]
  __int64 v99; // [rsp+28h] [rbp-228h]
  __int64 v100; // [rsp+28h] [rbp-228h]
  __int64 v101; // [rsp+28h] [rbp-228h]
  __int64 v102; // [rsp+28h] [rbp-228h]
  char v103[8]; // [rsp+30h] [rbp-220h] BYREF
  unsigned __int64 v104; // [rsp+38h] [rbp-218h]
  char v105; // [rsp+4Ch] [rbp-204h]
  _BYTE v106[64]; // [rsp+50h] [rbp-200h] BYREF
  __m128i *v107; // [rsp+90h] [rbp-1C0h]
  __int64 v108; // [rsp+98h] [rbp-1B8h]
  __int8 *v109; // [rsp+A0h] [rbp-1B0h]
  char v110[8]; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned __int64 v111; // [rsp+B8h] [rbp-198h]
  char v112; // [rsp+CCh] [rbp-184h]
  _BYTE v113[64]; // [rsp+D0h] [rbp-180h] BYREF
  unsigned __int64 v114; // [rsp+110h] [rbp-140h]
  unsigned __int64 v115; // [rsp+118h] [rbp-138h]
  unsigned __int64 v116; // [rsp+120h] [rbp-130h]
  _QWORD v117[3]; // [rsp+130h] [rbp-120h] BYREF
  char v118; // [rsp+14Ch] [rbp-104h]
  const __m128i *v119; // [rsp+190h] [rbp-C0h]
  const __m128i *v120; // [rsp+198h] [rbp-B8h]
  char v121[8]; // [rsp+1A8h] [rbp-A8h] BYREF
  unsigned __int64 v122; // [rsp+1B0h] [rbp-A0h]
  char v123; // [rsp+1C4h] [rbp-8Ch]
  const __m128i *v124; // [rsp+208h] [rbp-48h]
  const __m128i *v125; // [rsp+210h] [rbp-40h]

  *(_DWORD *)(a1 + 632) = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a1 + 200) = *(_QWORD *)(a1 + 216);
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 232);
  *(_QWORD *)(a1 + 192) = *(_QWORD *)(a1 + 240);
  v2 = *(_QWORD *)(a2 + 80);
  if ( v2 )
    v2 -= 24;
  sub_27A4830(v117, v2);
  v3 = v106;
  v4 = v103;
  sub_C8CD80((__int64)v103, (__int64)v106, (__int64)v117, v5, v6, v7);
  v10 = v120;
  v11 = v119;
  v107 = 0;
  v108 = 0;
  v12 = a1;
  v109 = 0;
  v13 = (char *)v120 - (char *)v119;
  if ( v120 == v119 )
  {
    v13 = 0;
    v14 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_137;
    v14 = (__m128i *)sub_22077B0((char *)v120 - (char *)v119);
    v10 = v120;
    v11 = v119;
    v12 = a1;
  }
  v107 = v14;
  v108 = (__int64)v14;
  v109 = &v14->m128i_i8[v13];
  if ( v11 == v10 )
  {
    v15 = (__int64)v14;
  }
  else
  {
    v15 = (__int64)v14->m128i_i64 + (char *)v10 - (char *)v11;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v11);
        v14[1] = _mm_loadu_si128(v11 + 1);
      }
      v14 += 2;
      v11 += 2;
    }
    while ( v14 != (__m128i *)v15 );
  }
  v4 = v110;
  v94 = v12;
  v3 = v113;
  v108 = v15;
  sub_C8CD80((__int64)v110, (__int64)v113, (__int64)v121, v15, v8, v9);
  v16 = v125;
  v17 = v124;
  v114 = 0;
  v115 = 0;
  v18 = v94;
  v116 = 0;
  v19 = (char *)v125 - (char *)v124;
  if ( v125 == v124 )
  {
    v21 = 0;
    goto LABEL_14;
  }
  if ( v19 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_137:
    sub_4261EA(v4, v3, v11);
  v20 = sub_22077B0((char *)v125 - (char *)v124);
  v17 = v124;
  v18 = v94;
  v21 = v20;
  v16 = v125;
LABEL_14:
  v114 = v21;
  v115 = v21;
  v116 = v21 + v19;
  if ( v16 == v17 )
  {
    v23 = (__m128i *)v21;
  }
  else
  {
    v22 = (__m128i *)v21;
    v23 = (__m128i *)(v21 + (char *)v16 - (char *)v17);
    do
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v17);
        v22[1] = _mm_loadu_si128(v17 + 1);
      }
      v22 += 2;
      v17 += 2;
    }
    while ( v23 != v22 );
  }
  v115 = (unsigned __int64)v23;
  v24 = v18;
  v95 = 0;
  v92 = v18 + 264;
  while ( 1 )
  {
    v25 = (unsigned __int64)v107;
    if ( (__m128i *)(v108 - (_QWORD)v107) != (__m128i *)((char *)v23 - v21) )
      goto LABEL_21;
    if ( v107 == (__m128i *)v108 )
      break;
    v61 = v21;
    while ( *(_QWORD *)v25 == *(_QWORD *)v61 )
    {
      v62 = *(_BYTE *)(v25 + 24);
      if ( v62 != *(_BYTE *)(v61 + 24) || v62 && *(_DWORD *)(v25 + 16) != *(_DWORD *)(v61 + 16) )
        break;
      v25 += 32LL;
      v61 += 32LL;
      if ( v108 == v25 )
        goto LABEL_70;
    }
LABEL_21:
    v26 = *(_DWORD *)(v24 + 288);
    ++v95;
    v27 = *(_QWORD *)(v108 - 32);
    if ( v26 )
    {
      v28 = *(_QWORD *)(v24 + 272);
      v29 = 1;
      v30 = 0;
      v31 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v32 = (_QWORD *)(v28 + 16LL * v31);
      v33 = *v32;
      if ( v27 == *v32 )
      {
LABEL_23:
        v34 = v32 + 1;
        goto LABEL_24;
      }
      while ( v33 != -4096 )
      {
        if ( !v30 && v33 == -8192 )
          v30 = v32;
        v31 = (v26 - 1) & (v29 + v31);
        v32 = (_QWORD *)(v28 + 16LL * v31);
        v33 = *v32;
        if ( v27 == *v32 )
          goto LABEL_23;
        ++v29;
      }
      if ( !v30 )
        v30 = v32;
      v51 = *(_DWORD *)(v24 + 280);
      ++*(_QWORD *)(v24 + 264);
      v52 = v51 + 1;
      if ( 4 * (v51 + 1) < 3 * v26 )
      {
        if ( v26 - *(_DWORD *)(v24 + 284) - v52 <= v26 >> 3 )
        {
          sub_A429D0(v92, v26);
          v78 = *(_DWORD *)(v24 + 288);
          if ( !v78 )
          {
LABEL_143:
            ++*(_DWORD *)(v24 + 280);
            BUG();
          }
          v79 = v78 - 1;
          v80 = *(_QWORD *)(v24 + 272);
          v81 = 0;
          v82 = v79 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v83 = 1;
          v52 = *(_DWORD *)(v24 + 280) + 1;
          v30 = (_QWORD *)(v80 + 16LL * v82);
          v84 = *v30;
          if ( v27 != *v30 )
          {
            while ( v84 != -4096 )
            {
              if ( !v81 && v84 == -8192 )
                v81 = v30;
              v82 = v79 & (v83 + v82);
              v30 = (_QWORD *)(v80 + 16LL * v82);
              v84 = *v30;
              if ( v27 == *v30 )
                goto LABEL_48;
              ++v83;
            }
            if ( v81 )
              v30 = v81;
          }
        }
        goto LABEL_48;
      }
    }
    else
    {
      ++*(_QWORD *)(v24 + 264);
    }
    sub_A429D0(v92, 2 * v26);
    v71 = *(_DWORD *)(v24 + 288);
    if ( !v71 )
      goto LABEL_143;
    v72 = v71 - 1;
    v73 = *(_QWORD *)(v24 + 272);
    v74 = (v71 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v52 = *(_DWORD *)(v24 + 280) + 1;
    v30 = (_QWORD *)(v73 + 16LL * v74);
    v75 = *v30;
    if ( v27 != *v30 )
    {
      v76 = 1;
      v77 = 0;
      while ( v75 != -4096 )
      {
        if ( v75 == -8192 && !v77 )
          v77 = v30;
        v74 = v72 & (v76 + v74);
        v30 = (_QWORD *)(v73 + 16LL * v74);
        v75 = *v30;
        if ( v27 == *v30 )
          goto LABEL_48;
        ++v76;
      }
      if ( v77 )
        v30 = v77;
    }
LABEL_48:
    *(_DWORD *)(v24 + 280) = v52;
    if ( *v30 != -4096 )
      --*(_DWORD *)(v24 + 284);
    *v30 = v27;
    v34 = v30 + 1;
    *((_DWORD *)v30 + 2) = 0;
LABEL_24:
    v35 = v27 + 48;
    v36 = 0;
    *v34 = v95;
    v37 = *(_QWORD *)(v27 + 56);
    if ( v27 + 48 != v37 )
    {
      while ( 1 )
      {
        v42 = *(_DWORD *)(v24 + 288);
        v43 = v37 - 24;
        if ( !v37 )
          v43 = 0;
        ++v36;
        if ( !v42 )
          break;
        v38 = *(_QWORD *)(v24 + 272);
        v39 = (v42 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v40 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( v43 == *v40 )
        {
LABEL_27:
          *((_DWORD *)v40 + 2) = v36;
          v37 = *(_QWORD *)(v37 + 8);
          if ( v35 == v37 )
            goto LABEL_37;
        }
        else
        {
          v90 = 1;
          v49 = 0;
          while ( v41 != -4096 )
          {
            if ( v41 == -8192 && !v49 )
              v49 = v40;
            v39 = (v42 - 1) & (v90 + v39);
            v40 = (__int64 *)(v38 + 16LL * v39);
            v41 = *v40;
            if ( v43 == *v40 )
              goto LABEL_27;
            ++v90;
          }
          if ( !v49 )
            v49 = v40;
          v53 = *(_DWORD *)(v24 + 280);
          ++*(_QWORD *)(v24 + 264);
          v48 = v53 + 1;
          if ( 4 * v48 < 3 * v42 )
          {
            if ( v42 - *(_DWORD *)(v24 + 284) - v48 <= v42 >> 3 )
            {
              v91 = v35;
              sub_A429D0(v92, v42);
              v54 = *(_DWORD *)(v24 + 288);
              if ( !v54 )
                goto LABEL_143;
              v55 = v54 - 1;
              v56 = *(_QWORD *)(v24 + 272);
              v57 = 0;
              v58 = v55 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
              v35 = v91;
              v59 = 1;
              v48 = *(_DWORD *)(v24 + 280) + 1;
              v49 = (__int64 *)(v56 + 16LL * v58);
              v60 = *v49;
              if ( v43 != *v49 )
              {
                while ( v60 != -4096 )
                {
                  if ( !v57 && v60 == -8192 )
                    v57 = v49;
                  v58 = v55 & (v59 + v58);
                  v49 = (__int64 *)(v56 + 16LL * v58);
                  v60 = *v49;
                  if ( v43 == *v49 )
                    goto LABEL_34;
                  ++v59;
                }
                if ( v57 )
                  v49 = v57;
              }
            }
            goto LABEL_34;
          }
LABEL_32:
          v89 = v35;
          sub_A429D0(v92, 2 * v42);
          v44 = *(_DWORD *)(v24 + 288);
          if ( !v44 )
            goto LABEL_143;
          v45 = v44 - 1;
          v46 = *(_QWORD *)(v24 + 272);
          v35 = v89;
          v47 = v45 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v48 = *(_DWORD *)(v24 + 280) + 1;
          v49 = (__int64 *)(v46 + 16LL * v47);
          v50 = *v49;
          if ( v43 != *v49 )
          {
            v85 = 1;
            v86 = 0;
            while ( v50 != -4096 )
            {
              if ( !v86 && v50 == -8192 )
                v86 = v49;
              v47 = v45 & (v85 + v47);
              v49 = (__int64 *)(v46 + 16LL * v47);
              v50 = *v49;
              if ( v43 == *v49 )
                goto LABEL_34;
              ++v85;
            }
            if ( v86 )
              v49 = v86;
          }
LABEL_34:
          *(_DWORD *)(v24 + 280) = v48;
          if ( *v49 != -4096 )
            --*(_DWORD *)(v24 + 284);
          *v49 = v43;
          *((_DWORD *)v49 + 2) = 0;
          *((_DWORD *)v49 + 2) = v36;
          v37 = *(_QWORD *)(v37 + 8);
          if ( v35 == v37 )
            goto LABEL_37;
        }
      }
      ++*(_QWORD *)(v24 + 264);
      goto LABEL_32;
    }
LABEL_37:
    sub_23EC7E0((__int64)v103);
    v21 = v114;
    v23 = (__m128i *)v115;
  }
LABEL_70:
  v63 = v24;
  if ( v21 )
  {
    j_j___libc_free_0(v21);
    v63 = v24;
  }
  if ( !v112 )
  {
    v99 = v63;
    _libc_free(v111);
    v63 = v99;
  }
  if ( v107 )
  {
    v96 = v63;
    j_j___libc_free_0((unsigned __int64)v107);
    v63 = v96;
  }
  if ( !v105 )
  {
    v102 = v63;
    _libc_free(v104);
    v63 = v102;
  }
  if ( v124 )
  {
    v97 = v63;
    j_j___libc_free_0((unsigned __int64)v124);
    v63 = v97;
  }
  if ( !v123 )
  {
    v101 = v63;
    _libc_free(v122);
    v63 = v101;
  }
  if ( v119 )
  {
    v98 = v63;
    j_j___libc_free_0((unsigned __int64)v119);
    v63 = v98;
  }
  if ( !v118 )
  {
    v100 = v63;
    _libc_free(v117[1]);
    v63 = v100;
  }
  v64 = 0;
  v65 = 0;
  v66 = v63;
  while ( (_DWORD)qword_4FFC228 == -1 || ++v64 < (int)qword_4FFC228 )
  {
    v67 = sub_27A9280(v66, a2);
    if ( !(HIDWORD(v67) + (_DWORD)v67) )
      break;
    if ( HIDWORD(v67) )
      sub_278EBA0(v66, a2, HIDWORD(v67), v68, v69, v70);
    v65 = 1;
  }
  return v65;
}
