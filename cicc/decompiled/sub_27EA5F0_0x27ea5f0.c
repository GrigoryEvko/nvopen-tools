// Function: sub_27EA5F0
// Address: 0x27ea5f0
//
__int64 __fastcall sub_27EA5F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8,
        __int128 a9,
        __int128 a10)
{
  __int64 v11; // rdx
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rbx
  _QWORD *v18; // r15
  void (__fastcall *v19)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  __m128i v22; // xmm1
  __int64 v23; // rax
  __int64 v24; // rax
  bool v25; // dl
  __int64 v26; // rax
  _QWORD *v27; // rbx
  _BYTE *v28; // r15
  size_t v29; // r12
  _QWORD *v30; // rax
  _QWORD *v31; // r12
  _QWORD *v32; // rbx
  unsigned __int64 v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r12
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // rsi
  __int64 v52; // rbx
  __int64 v53; // r13
  __int64 v54; // rsi
  __int64 v55; // rdx
  unsigned int v56; // eax
  _QWORD *v57; // rax
  __int64 v58; // rbx
  __int64 v59; // r13
  char v60; // r12
  __int64 v61; // r15
  _QWORD *v62; // rax
  _QWORD *v63; // rdx
  unsigned int v64; // eax
  __int64 v65; // rdx
  _QWORD *v67; // r12
  _QWORD *v68; // rbx
  unsigned __int64 v69; // rdx
  _QWORD *v70; // rax
  _QWORD *v71; // rdi
  __int64 v72; // rsi
  __int64 v73; // rcx
  __int64 v74; // rax
  _QWORD *v75; // rdx
  __int64 v76; // rdi
  char v77; // al
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rdx
  _QWORD *v81; // rax
  _QWORD *v82; // rdx
  __int64 v83; // rax
  unsigned __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // r8
  __int64 v87; // r9
  _QWORD *v88; // rdx
  __int64 v89; // rcx
  _QWORD *v90; // rax
  _QWORD *v91; // rsi
  _QWORD *v92; // rdx
  _QWORD *v93; // rax
  __int64 v94; // rdx
  __int64 *v95; // rax
  _QWORD *v96; // rdi
  char v97; // al
  __int64 v98; // rbx
  __int64 i; // r12
  __int64 v100; // rdi
  __int64 v101; // [rsp+8h] [rbp-98h]
  char v102; // [rsp+8h] [rbp-98h]
  unsigned __int8 v103; // [rsp+17h] [rbp-89h]
  __int64 v104; // [rsp+18h] [rbp-88h]
  size_t v105; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v106[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v107[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v108; // [rsp+50h] [rbp-50h]
  __int64 v109; // [rsp+58h] [rbp-48h]
  __int64 v110; // [rsp+60h] [rbp-40h]

  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 32) = a6;
  *(_QWORD *)(a1 + 40) = a7;
  v11 = *a8;
  *a8 = 0;
  v12 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v11;
  if ( v12 )
  {
    sub_FFCE90(v12, a2, v11, a4, a5, a6);
    sub_FFD870(v12, a2, v13, v14, v15, v16);
    sub_FFBC40(v12, a2);
    v17 = *(_QWORD **)(v12 + 680);
    v18 = *(_QWORD **)(v12 + 672);
    if ( v17 != v18 )
    {
      do
      {
        v19 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v18[7];
        *v18 = &unk_49E5048;
        if ( v19 )
          v19(v18 + 5, v18 + 5, 3);
        *v18 = &unk_49DB368;
        v20 = v18[3];
        if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
          sub_BD60C0(v18 + 1);
        v18 += 9;
      }
      while ( v17 != v18 );
      v18 = *(_QWORD **)(v12 + 672);
    }
    if ( v18 )
      j_j___libc_free_0((unsigned __int64)v18);
    if ( *(_BYTE *)(v12 + 596) )
    {
      v21 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == v12 + 16 )
      {
LABEL_15:
        j_j___libc_free_0(v12);
        goto LABEL_16;
      }
    }
    else
    {
      _libc_free(*(_QWORD *)(v12 + 576));
      v21 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == v12 + 16 )
        goto LABEL_15;
    }
    _libc_free(v21);
    goto LABEL_15;
  }
LABEL_16:
  v22 = _mm_loadu_si128((const __m128i *)&a10);
  v23 = *(_QWORD *)a1;
  *(__m128i *)(a1 + 56) = _mm_loadu_si128((const __m128i *)&a9);
  *(__m128i *)(a1 + 72) = v22;
  v24 = sub_B6AC80(*(_QWORD *)(v23 + 40), 153);
  v25 = 0;
  if ( v24 )
    v25 = *(_QWORD *)(v24 + 16) != 0;
  v26 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 89) = v25;
  v106[0] = (unsigned __int64)v107;
  v27 = *(_QWORD **)(v26 + 40);
  v28 = (_BYTE *)v27[29];
  v29 = v27[30];
  if ( v28 == 0 && &v28[v29] != 0 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v105 = v27[30];
  if ( v29 > 0xF )
  {
    v106[0] = sub_22409D0((__int64)v106, &v105, 0);
    v96 = (_QWORD *)v106[0];
    v107[0] = v105;
    goto LABEL_156;
  }
  if ( v29 != 1 )
  {
    if ( !v29 )
    {
      v30 = v107;
      goto LABEL_24;
    }
    v96 = v107;
LABEL_156:
    memcpy(v96, v28, v29);
    v29 = v105;
    v30 = (_QWORD *)v106[0];
    goto LABEL_24;
  }
  LOBYTE(v107[0]) = *v28;
  v30 = v107;
LABEL_24:
  v106[1] = v29;
  *((_BYTE *)v30 + v29) = 0;
  v108 = v27[33];
  v109 = v27[34];
  v110 = v27[35];
  if ( (unsigned int)(v108 - 42) > 1 )
    goto LABEL_86;
  v31 = sub_C52410();
  v32 = v31 + 1;
  v33 = sub_C959E0();
  v34 = (_QWORD *)v31[2];
  if ( v34 )
  {
    v35 = v31 + 1;
    do
    {
      while ( 1 )
      {
        v36 = v34[2];
        v37 = v34[3];
        if ( v33 <= v34[4] )
          break;
        v34 = (_QWORD *)v34[3];
        if ( !v37 )
          goto LABEL_30;
      }
      v35 = v34;
      v34 = (_QWORD *)v34[2];
    }
    while ( v36 );
LABEL_30:
    if ( v32 != v35 && v33 >= v35[4] )
      v32 = v35;
  }
  if ( v32 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_41;
  v42 = v32[7];
  v40 = (__int64)(v32 + 6);
  if ( !v42 )
    goto LABEL_41;
  v33 = (unsigned int)dword_4FFDBA8;
  v43 = v32 + 6;
  do
  {
    while ( 1 )
    {
      v39 = *(_QWORD *)(v42 + 16);
      v38 = *(_QWORD *)(v42 + 24);
      if ( *(_DWORD *)(v42 + 32) >= dword_4FFDBA8 )
        break;
      v42 = *(_QWORD *)(v42 + 24);
      if ( !v38 )
        goto LABEL_39;
    }
    v43 = (_QWORD *)v42;
    v42 = *(_QWORD *)(v42 + 16);
  }
  while ( v39 );
LABEL_39:
  if ( (_QWORD *)v40 == v43 || dword_4FFDBA8 < *((_DWORD *)v43 + 8) || !*((_DWORD *)v43 + 9) )
  {
LABEL_41:
    if ( (_QWORD *)v106[0] != v107 )
    {
      v33 = v107[0] + 1LL;
      j_j___libc_free_0(v106[0]);
    }
    *(_DWORD *)(a1 + 416) = 1;
  }
  else
  {
LABEL_86:
    if ( (_QWORD *)v106[0] != v107 )
      j_j___libc_free_0(v106[0]);
    v67 = sub_C52410();
    v68 = v67 + 1;
    v69 = sub_C959E0();
    v70 = (_QWORD *)v67[2];
    if ( v70 )
    {
      v71 = v67 + 1;
      do
      {
        while ( 1 )
        {
          v72 = v70[2];
          v73 = v70[3];
          if ( v69 <= v70[4] )
            break;
          v70 = (_QWORD *)v70[3];
          if ( !v73 )
            goto LABEL_93;
        }
        v71 = v70;
        v70 = (_QWORD *)v70[2];
      }
      while ( v72 );
LABEL_93:
      if ( v68 != v71 && v69 >= v71[4] )
        v68 = v71;
    }
    if ( v68 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_104;
    v74 = v68[7];
    v40 = (__int64)(v68 + 6);
    if ( !v74 )
      goto LABEL_104;
    v39 = (unsigned int)dword_4FFDBA8;
    v75 = v68 + 6;
    do
    {
      while ( 1 )
      {
        v76 = *(_QWORD *)(v74 + 16);
        v33 = *(_QWORD *)(v74 + 24);
        if ( *(_DWORD *)(v74 + 32) >= dword_4FFDBA8 )
          break;
        v74 = *(_QWORD *)(v74 + 24);
        if ( !v33 )
          goto LABEL_102;
      }
      v75 = (_QWORD *)v74;
      v74 = *(_QWORD *)(v74 + 16);
    }
    while ( v76 );
LABEL_102:
    if ( v75 == (_QWORD *)v40
      || dword_4FFDBA8 < *((_DWORD *)v75 + 8)
      || (v38 = *((unsigned int *)v75 + 9), !(_DWORD)v38) )
    {
LABEL_104:
      v33 = 18;
      if ( (unsigned __int8)sub_B2D610(*(_QWORD *)a1, 18) )
        *(_DWORD *)(a1 + 416) = 3;
      else
        *(_DWORD *)(a1 + 416) = *(_DWORD *)(a1 + 420);
    }
    else
    {
      *(_DWORD *)(a1 + 416) = qword_4FFDC28;
    }
  }
  v44 = sub_FFD350(*(_QWORD *)(a1 + 48), v33, v38, v39, v40, v41);
  ++*(_QWORD *)(a1 + 256);
  v48 = v44;
  v104 = a1 + 256;
  if ( *(_BYTE *)(a1 + 284) )
    goto LABEL_49;
  v49 = 4 * (*(_DWORD *)(a1 + 276) - *(_DWORD *)(a1 + 280));
  v50 = *(unsigned int *)(a1 + 272);
  if ( v49 < 0x20 )
    v49 = 32;
  if ( (unsigned int)v50 <= v49 )
  {
    memset(*(void **)(a1 + 264), -1, 8 * v50);
LABEL_49:
    *(_QWORD *)(a1 + 276) = 0;
    goto LABEL_50;
  }
  sub_C8C990(v104, v33);
LABEL_50:
  v51 = *(_QWORD *)a1;
  v52 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v53 = *(_QWORD *)a1 + 72LL;
  if ( v52 != v53 )
  {
    while ( 1 )
    {
      if ( v52 )
      {
        v54 = v52 - 24;
        v55 = (unsigned int)(*(_DWORD *)(v52 + 20) + 1);
        v56 = *(_DWORD *)(v52 + 20) + 1;
      }
      else
      {
        v54 = 0;
        v55 = 0;
        v56 = 0;
      }
      if ( v56 < *(_DWORD *)(v48 + 32) && *(_QWORD *)(*(_QWORD *)(v48 + 24) + 8 * v55) )
        goto LABEL_53;
      if ( !*(_BYTE *)(a1 + 284) )
        goto LABEL_106;
      v57 = *(_QWORD **)(a1 + 264);
      v45 = *(unsigned int *)(a1 + 276);
      v55 = (__int64)&v57[v45];
      if ( v57 != (_QWORD *)v55 )
      {
        while ( v54 != *v57 )
        {
          if ( (_QWORD *)v55 == ++v57 )
            goto LABEL_61;
        }
        goto LABEL_53;
      }
LABEL_61:
      if ( (unsigned int)v45 < *(_DWORD *)(a1 + 272) )
      {
        v45 = (unsigned int)(v45 + 1);
        *(_DWORD *)(a1 + 276) = v45;
        *(_QWORD *)v55 = v54;
        ++*(_QWORD *)(a1 + 256);
        v52 = *(_QWORD *)(v52 + 8);
        if ( v52 == v53 )
        {
LABEL_63:
          v51 = *(_QWORD *)a1;
          break;
        }
      }
      else
      {
LABEL_106:
        sub_C8CC70(v104, v54, v55, v45, v46, v47);
LABEL_53:
        v52 = *(_QWORD *)(v52 + 8);
        if ( v52 == v53 )
          goto LABEL_63;
      }
    }
  }
  if ( !(_BYTE)qword_4FFD988 )
  {
    sub_27DC8E0(a1, v51);
    v51 = *(_QWORD *)a1;
  }
  v58 = *(_QWORD *)(v51 + 80);
  v59 = v51 + 72;
  v103 = 0;
  if ( v51 + 72 == v58 )
    goto LABEL_78;
  while ( 2 )
  {
    v60 = 0;
    do
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v61 = 0;
          if ( v58 )
            v61 = v58 - 24;
          if ( *(_BYTE *)(a1 + 284) )
            break;
          v51 = v61;
          if ( !sub_C8CA60(v104, v61) )
            goto LABEL_109;
          v58 = *(_QWORD *)(v58 + 8);
          if ( v59 == v58 )
            goto LABEL_76;
        }
        v62 = *(_QWORD **)(a1 + 264);
        v63 = &v62[*(unsigned int *)(a1 + 276)];
        if ( v62 != v63 )
        {
          while ( v61 != *v62 )
          {
            if ( v63 == ++v62 )
              goto LABEL_109;
          }
          goto LABEL_75;
        }
LABEL_109:
        while ( 1 )
        {
          v51 = v61;
          v77 = sub_27E95E0(a1, v61);
          if ( !v77 )
            break;
          *(_BYTE *)(a1 + 88) = 1;
          v60 = v77;
        }
        v78 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
        if ( v78 )
          v78 -= 24;
        if ( v61 == v78 )
          goto LABEL_75;
        v79 = *(_QWORD *)(a1 + 48);
        if ( *(_BYTE *)(v79 + 560) )
        {
          v80 = *(unsigned int *)(v79 + 588);
          if ( (_DWORD)v80 != *(_DWORD *)(v79 + 592) )
          {
            if ( *(_BYTE *)(v79 + 596) )
            {
              v81 = *(_QWORD **)(v79 + 576);
              v82 = &v81[v80];
              if ( v81 != v82 )
              {
                while ( v61 != *v81 )
                {
                  if ( v82 == ++v81 )
                    goto LABEL_125;
                }
                goto LABEL_75;
              }
            }
            else
            {
              v51 = v61;
              if ( sub_C8CA60(v79 + 568, v61) )
                goto LABEL_75;
            }
          }
        }
LABEL_125:
        v83 = *(_QWORD *)(v61 + 16);
        if ( !v83 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v83 + 24) - 30) > 0xAu )
        {
          v83 = *(_QWORD *)(v83 + 8);
          if ( !v83 )
            goto LABEL_145;
        }
        v84 = *(_QWORD *)(v61 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v84 == v61 + 48 || !v84 || (unsigned int)*(unsigned __int8 *)(v84 - 24) - 30 > 0xA )
LABEL_179:
          BUG();
        if ( *(_BYTE *)(v84 - 24) != 31 || (*(_DWORD *)(v84 - 20) & 0x7FFFFFF) != 1 )
          goto LABEL_75;
        v51 = 1;
        v101 = *(_QWORD *)(v84 - 56);
        v85 = sub_AA5030(v61, 1);
        if ( !v85 )
          goto LABEL_179;
        if ( (unsigned int)*(unsigned __int8 *)(v85 - 24) - 30 > 0xA )
          goto LABEL_75;
        if ( *(_BYTE *)(a1 + 124) )
        {
          v88 = *(_QWORD **)(a1 + 104);
          v89 = (__int64)&v88[*(unsigned int *)(a1 + 116)];
          if ( v88 == (_QWORD *)v89 )
            goto LABEL_160;
          v90 = *(_QWORD **)(a1 + 104);
          while ( v61 != *v90 )
          {
            if ( (_QWORD *)v89 == ++v90 )
              goto LABEL_142;
          }
          v58 = *(_QWORD *)(v58 + 8);
          if ( v59 == v58 )
            goto LABEL_76;
        }
        else
        {
          v51 = v61;
          if ( !sub_C8CA60(a1 + 96, v61) )
          {
            if ( *(_BYTE *)(a1 + 124) )
            {
              v88 = *(_QWORD **)(a1 + 104);
              v90 = &v88[*(unsigned int *)(a1 + 116)];
              if ( v90 != v88 )
              {
LABEL_142:
                do
                {
                  v89 = v101;
                  if ( v101 == *v88 )
                    goto LABEL_75;
                  ++v88;
                }
                while ( v90 != v88 );
              }
LABEL_160:
              v51 = *(_QWORD *)(a1 + 48);
              v97 = sub_F5F040(v61, v51, (__int64)v88, v89, v86, v87);
              if ( v97 )
              {
                v51 = v61;
                v102 = v97;
                sub_22C2BE0(*(_QWORD *)(a1 + 32), v61);
                *(_BYTE *)(a1 + 88) = 1;
                v60 = v102;
              }
              goto LABEL_75;
            }
            v51 = v101;
            if ( !sub_C8CA60(a1 + 96, v101) )
              goto LABEL_160;
          }
LABEL_75:
          v58 = *(_QWORD *)(v58 + 8);
          if ( v59 == v58 )
            goto LABEL_76;
        }
      }
LABEL_145:
      if ( *(_BYTE *)(a1 + 124) )
      {
        v91 = *(_QWORD **)(a1 + 104);
        v92 = &v91[*(unsigned int *)(a1 + 116)];
        v93 = v91;
        if ( v91 != v92 )
        {
          while ( v61 != *v93 )
          {
            if ( v92 == ++v93 )
              goto LABEL_151;
          }
          v94 = (unsigned int)(*(_DWORD *)(a1 + 116) - 1);
          *(_DWORD *)(a1 + 116) = v94;
          *v93 = v91[v94];
          ++*(_QWORD *)(a1 + 96);
        }
      }
      else
      {
        v95 = sub_C8CA60(a1 + 96, v61);
        if ( v95 )
        {
          *v95 = -2;
          ++*(_DWORD *)(a1 + 120);
          ++*(_QWORD *)(a1 + 96);
        }
      }
LABEL_151:
      v60 = 1;
      sub_22C2BE0(*(_QWORD *)(a1 + 32), v61);
      v51 = *(_QWORD *)(a1 + 48);
      sub_F34560(v61, v51, 0);
      *(_BYTE *)(a1 + 88) = 1;
      v58 = *(_QWORD *)(v58 + 8);
    }
    while ( v59 != v58 );
LABEL_76:
    if ( v60 )
    {
      v51 = *(_QWORD *)a1;
      v103 = v60;
      v58 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
      v59 = *(_QWORD *)a1 + 72LL;
      if ( v59 == v58 )
        goto LABEL_78;
      continue;
    }
    break;
  }
  if ( v103 )
  {
    v98 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    for ( i = *(_QWORD *)a1 + 72LL; i != v98; v98 = *(_QWORD *)(v98 + 8) )
    {
      v100 = v98 - 24;
      if ( !v98 )
        v100 = 0;
      sub_F3F2F0(v100, v51);
    }
  }
LABEL_78:
  ++*(_QWORD *)(a1 + 96);
  if ( *(_BYTE *)(a1 + 124) )
  {
LABEL_83:
    *(_QWORD *)(a1 + 116) = 0;
  }
  else
  {
    v64 = 4 * (*(_DWORD *)(a1 + 116) - *(_DWORD *)(a1 + 120));
    v65 = *(unsigned int *)(a1 + 112);
    if ( v64 < 0x20 )
      v64 = 32;
    if ( (unsigned int)v65 <= v64 )
    {
      memset(*(void **)(a1 + 104), -1, 8 * v65);
      goto LABEL_83;
    }
    sub_C8C990(a1 + 96, v51);
  }
  return v103;
}
