// Function: sub_35B7220
// Address: 0x35b7220
//
__int64 __fastcall sub_35B7220(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r12d
  int v11; // r14d
  unsigned int v12; // r15d
  int v13; // eax
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // edx
  __int64 *v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned int v24; // r14d
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // r13
  __int64 v30; // r13
  __int64 *v31; // rsi
  __int64 v32; // r15
  __int64 v33; // r14
  float v34; // xmm0_4
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int16 *v37; // rax
  unsigned __int16 *v38; // r11
  __int64 v39; // rcx
  __int64 v40; // rdi
  __int64 v41; // r14
  _QWORD **v42; // rdi
  _QWORD **v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // r12
  unsigned __int64 v46; // r13
  int v48; // ecx
  int v49; // edx
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // r9
  __int64 v56; // rdx
  __int64 (*v57)(void); // rsi
  __int64 v58; // rax
  int v59; // eax
  _QWORD **v60; // rax
  _QWORD *v61; // rdi
  __int64 *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 v68; // rcx
  __int64 v69; // rdx
  int v70; // edx
  _QWORD **v71; // rdx
  _QWORD *v72; // rsi
  __int64 v73; // r12
  _QWORD **v74; // rcx
  _QWORD **v75; // rsi
  _QWORD **v76; // rdx
  __int64 v77; // rax
  __int64 *v78; // rax
  unsigned int *v79; // [rsp+8h] [rbp-218h]
  unsigned int v80; // [rsp+14h] [rbp-20Ch]
  __int64 v81; // [rsp+20h] [rbp-200h]
  __int16 *v82; // [rsp+30h] [rbp-1F0h]
  unsigned int *v83; // [rsp+38h] [rbp-1E8h]
  unsigned int *v85; // [rsp+50h] [rbp-1D0h] BYREF
  __int64 v86; // [rsp+58h] [rbp-1C8h]
  _BYTE v87[32]; // [rsp+60h] [rbp-1C0h] BYREF
  unsigned __int16 *v88; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v89; // [rsp+88h] [rbp-198h]
  char v90; // [rsp+98h] [rbp-188h] BYREF
  __int64 v91; // [rsp+B8h] [rbp-168h]
  int v92; // [rsp+C8h] [rbp-158h]
  __int64 *v93; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v94; // [rsp+D8h] [rbp-148h]
  char v95; // [rsp+E0h] [rbp-140h] BYREF
  void *v96; // [rsp+120h] [rbp-100h] BYREF
  __int64 v97; // [rsp+128h] [rbp-F8h]
  __int64 v98; // [rsp+130h] [rbp-F0h]
  __int64 v99; // [rsp+138h] [rbp-E8h]
  __int64 v100; // [rsp+140h] [rbp-E0h]
  __int64 v101; // [rsp+148h] [rbp-D8h]
  __int64 v102; // [rsp+150h] [rbp-D0h]
  __int64 v103; // [rsp+158h] [rbp-C8h]
  int v104; // [rsp+160h] [rbp-C0h]
  char v105; // [rsp+164h] [rbp-BCh]
  __int64 v106; // [rsp+168h] [rbp-B8h]
  __int64 v107; // [rsp+170h] [rbp-B0h]
  _BYTE *v108; // [rsp+178h] [rbp-A8h]
  __int64 v109; // [rsp+180h] [rbp-A0h]
  int v110; // [rsp+188h] [rbp-98h]
  char v111; // [rsp+18Ch] [rbp-94h]
  _BYTE v112[32]; // [rsp+190h] [rbp-90h] BYREF
  __int64 v113; // [rsp+1B0h] [rbp-70h]
  _BYTE *v114; // [rsp+1B8h] [rbp-68h]
  __int64 v115; // [rsp+1C0h] [rbp-60h]
  int v116; // [rsp+1C8h] [rbp-58h]
  char v117; // [rsp+1CCh] [rbp-54h]
  _BYTE v118[80]; // [rsp+1D0h] [rbp-50h] BYREF

  v3 = a2;
  v5 = *(_DWORD *)(a2 + 112);
  v6 = *(_QWORD *)(a1 + 224);
  v7 = *(_QWORD *)(a1 + 240);
  v85 = (unsigned int *)v87;
  v86 = 0x800000000LL;
  sub_34B8230((__int64)&v88, v5, v6, a1 + 248, v7);
  v10 = v92;
  v11 = -(int)v89;
  if ( -(int)v89 != v92 )
  {
    while ( 1 )
    {
      if ( v11 < 0 )
        v12 = v88[v89 + v11];
      else
        v12 = *(unsigned __int16 *)(v91 + 2LL * v11);
      v13 = sub_2E21680(*(_QWORD **)(a1 + 240), v3, v12);
      if ( !v13 )
        goto LABEL_64;
      if ( v13 == 1 )
      {
        v15 = (unsigned int)v86;
        v16 = (unsigned int)v86 + 1LL;
        if ( v16 > HIDWORD(v86) )
        {
          sub_C8D5F0((__int64)&v85, v87, v16, 4u, v8, v14);
          v15 = (unsigned int)v86;
        }
        v85[v15] = v12;
        LODWORD(v86) = v86 + 1;
      }
      v9 = (unsigned int)v92;
      if ( v92 > v11 && ++v11 >= 0 && v92 > v11 )
        break;
LABEL_14:
      if ( v10 == v11 )
        goto LABEL_15;
    }
    v17 = v11;
    while ( 1 )
    {
      v18 = *(unsigned __int16 *)(v91 + 2 * v17);
      v8 = (unsigned int)v17;
      v11 = v17;
      if ( (unsigned int)(v18 - 1) > 0x3FFFFFFE )
        goto LABEL_14;
      v37 = v88;
      v38 = &v88[v89];
      v39 = (2 * v89) >> 3;
      v40 = (2 * v89) >> 1;
      if ( v39 > 0 )
      {
        while ( v18 != *v37 )
        {
          if ( v18 == v37[1] )
          {
            ++v37;
            goto LABEL_35;
          }
          if ( v18 == v37[2] )
          {
            v37 += 2;
            goto LABEL_35;
          }
          if ( v18 == v37[3] )
          {
            v37 += 3;
            goto LABEL_35;
          }
          v37 += 4;
          if ( v37 == &v88[4 * v39] )
          {
            v40 = v38 - v37;
            goto LABEL_39;
          }
        }
        goto LABEL_35;
      }
LABEL_39:
      if ( v40 != 2 )
      {
        if ( v40 != 3 )
        {
          if ( v40 != 1 || v18 != *v37 )
            goto LABEL_14;
          goto LABEL_35;
        }
        if ( v18 == *v37 )
          goto LABEL_35;
        ++v37;
      }
      if ( v18 != *v37 )
      {
        v48 = v37[1];
        ++v37;
        if ( v18 != v48 )
          goto LABEL_14;
      }
LABEL_35:
      if ( v38 != v37 )
      {
        ++v17;
        v11 = v8 + 1;
        if ( v92 > (int)v17 )
          continue;
      }
      goto LABEL_14;
    }
  }
LABEL_15:
  v79 = &v85[(unsigned int)v86];
  if ( v79 == v85 )
    goto LABEL_78;
  v83 = v85;
  v19 = (__int64 *)&v95;
  do
  {
    v20 = *(_QWORD *)(a1 + 208);
    v21 = *v83;
    v94 = 0x800000000LL;
    v22 = *(_QWORD *)(v20 + 8);
    v23 = *(_QWORD *)(v20 + 56);
    v93 = v19;
    LODWORD(v21) = *(_DWORD *)(v22 + 24 * v21 + 16);
    v24 = v21 & 0xFFF;
    v82 = (__int16 *)(v23 + 2LL * ((unsigned int)v21 >> 12));
    v25 = v3;
    while ( 1 )
    {
      if ( !v82 )
      {
LABEL_82:
        v45 = v93;
        v46 = (unsigned __int64)&v93[(unsigned int)v94];
        if ( v93 == (__int64 *)v46 )
        {
LABEL_61:
          if ( (__int64 *)v46 != v19 )
            _libc_free(v46);
          v12 = *v83;
          goto LABEL_64;
        }
        while ( 1 )
        {
          v50 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 224) + 32LL)
                                + 4LL * (*(_DWORD *)(*v45 + 112) & 0x7FFFFFFF));
          if ( (_DWORD)v50 )
            break;
LABEL_59:
          if ( (__int64 *)v46 == ++v45 )
          {
            v46 = (unsigned __int64)v93;
            goto LABEL_61;
          }
        }
        v81 = *v45;
        sub_2E21040(*(_QWORD **)(a1 + 240), *v45, v50, v22, v8);
        v53 = *(_QWORD *)(a1 + 968);
        v54 = *(_QWORD *)(a1 + 224);
        v97 = v81;
        v96 = &unk_4A388F0;
        v55 = *(_QWORD *)(a1 + 232);
        v98 = a3;
        v56 = *(_QWORD *)(v53 + 32);
        v100 = v55;
        v99 = v56;
        v101 = v54;
        v57 = *(__int64 (**)(void))(**(_QWORD **)(v53 + 16) + 128LL);
        v58 = 0;
        if ( v57 != sub_2DAC790 )
        {
          v58 = v57();
          v56 = v99;
        }
        v102 = v58;
        v103 = a1 + 960;
        v59 = *(_DWORD *)(a3 + 8);
        v117 = 1;
        v105 = 0;
        v104 = v59;
        v107 = 0;
        v106 = a1 + 600;
        v108 = v112;
        v109 = 4;
        v110 = 0;
        v111 = 1;
        v113 = 0;
        v114 = v118;
        v115 = 4;
        v116 = 0;
        if ( !*(_BYTE *)(v56 + 36) )
          goto LABEL_48;
        v60 = *(_QWORD ***)(v56 + 16);
        v61 = &v60[*(unsigned int *)(v56 + 28)];
        v52 = *(unsigned int *)(v56 + 28);
        if ( v60 != v61 )
        {
          while ( *v60 != &v96 )
          {
            if ( v61 == ++v60 )
              goto LABEL_94;
          }
          goto LABEL_49;
        }
LABEL_94:
        if ( (unsigned int)v52 < *(_DWORD *)(v56 + 24) )
        {
          *(_DWORD *)(v56 + 28) = v52 + 1;
          *v61 = &v96;
          ++*(_QWORD *)(v56 + 8);
        }
        else
        {
LABEL_48:
          sub_C8CC70(v56 + 8, (__int64)&v96, v56, v51, v52, v55);
        }
LABEL_49:
        (*(void (__fastcall **)(_QWORD, void **))(**(_QWORD **)(a1 + 976) + 24LL))(*(_QWORD *)(a1 + 976), &v96);
        v41 = v99;
        v96 = &unk_4A388F0;
        if ( *(_BYTE *)(v99 + 36) )
        {
          v42 = *(_QWORD ***)(v99 + 16);
          v8 = (__int64)&v42[*(unsigned int *)(v99 + 28)];
          v43 = v42;
          if ( v42 != (_QWORD **)v8 )
          {
            while ( *v43 != &v96 )
            {
              if ( (_QWORD **)v8 == ++v43 )
                goto LABEL_55;
            }
            v44 = (unsigned int)(*(_DWORD *)(v99 + 28) - 1);
            *(_DWORD *)(v99 + 28) = v44;
            *v43 = v42[v44];
            ++*(_QWORD *)(v41 + 8);
          }
        }
        else
        {
          v62 = sub_C8CA60(v99 + 8, (__int64)&v96);
          if ( v62 )
          {
            *v62 = -2;
            ++*(_DWORD *)(v41 + 32);
            ++*(_QWORD *)(v41 + 8);
          }
        }
LABEL_55:
        if ( !v117 )
          _libc_free((unsigned __int64)v114);
        if ( !v111 )
          _libc_free((unsigned __int64)v108);
        goto LABEL_59;
      }
      v26 = sub_2E21610(*(_QWORD *)(a1 + 240), v25, v24);
      v29 = v26;
      if ( !*(_BYTE *)(v26 + 161) )
        sub_2E1AC90(v26, 0xFFFFFFFF, v27, v22, v8, v28);
      v9 = *(_QWORD *)(v29 + 112);
      v30 = v9 + 8LL * *(unsigned int *)(v29 + 120);
      if ( v9 != v30 )
        break;
LABEL_81:
      v49 = *v82++;
      v24 += v49;
      if ( !(_WORD)v49 )
        goto LABEL_82;
    }
    v31 = v19;
    v80 = v24;
    v32 = v9;
    while ( 1 )
    {
      v33 = *(_QWORD *)(v30 - 8);
      v34 = *(float *)(v33 + 116);
      if ( v34 == INFINITY || v34 > *(float *)(v25 + 116) )
        break;
      v35 = (unsigned int)v94;
      v22 = HIDWORD(v94);
      v36 = (unsigned int)v94 + 1LL;
      if ( v36 > HIDWORD(v94) )
      {
        sub_C8D5F0((__int64)&v93, v31, v36, 8u, v8, v9);
        v35 = (unsigned int)v94;
      }
      v30 -= 8;
      v93[v35] = v33;
      LODWORD(v94) = v94 + 1;
      if ( v32 == v30 )
      {
        v24 = v80;
        v19 = v31;
        goto LABEL_81;
      }
    }
    v3 = v25;
    v19 = v31;
    if ( v93 != v31 )
      _libc_free((unsigned __int64)v93);
    ++v83;
  }
  while ( v79 != v83 );
LABEL_78:
  if ( *(float *)(v3 + 116) == INFINITY )
  {
    v12 = -1;
    goto LABEL_64;
  }
  v63 = *(_QWORD *)(a1 + 968);
  v97 = v3;
  v64 = *(_QWORD *)(a1 + 224);
  v65 = *(_QWORD *)(a1 + 232);
  v96 = &unk_4A388F0;
  v98 = a3;
  v66 = *(_QWORD *)(v63 + 32);
  v100 = v65;
  v99 = v66;
  v101 = v64;
  v67 = *(_QWORD *)(v63 + 16);
  v68 = *(_QWORD *)(*(_QWORD *)v67 + 128LL);
  v69 = 0;
  if ( (__int64 (*)())v68 != sub_2DAC790 )
  {
    v69 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v68)(v67, v65, 0);
    v66 = v99;
  }
  v102 = v69;
  v103 = a1 + 960;
  v70 = *(_DWORD *)(a3 + 8);
  v117 = 1;
  v105 = 0;
  v104 = v70;
  v108 = v112;
  v71 = (_QWORD **)v118;
  v106 = a1 + 600;
  v107 = 0;
  v109 = 4;
  v110 = 0;
  v111 = 1;
  v113 = 0;
  v114 = v118;
  v115 = 4;
  v116 = 0;
  if ( !*(_BYTE *)(v66 + 36) )
    goto LABEL_118;
  v71 = *(_QWORD ***)(v66 + 16);
  v68 = *(unsigned int *)(v66 + 28);
  v72 = &v71[v68];
  if ( v71 == v72 )
  {
LABEL_116:
    if ( (unsigned int)v68 < *(_DWORD *)(v66 + 24) )
    {
      *(_DWORD *)(v66 + 28) = v68 + 1;
      *v72 = &v96;
      ++*(_QWORD *)(v66 + 8);
      goto LABEL_104;
    }
LABEL_118:
    sub_C8CC70(v66 + 8, (__int64)&v96, (__int64)v71, v68, v8, v9);
  }
  else
  {
    while ( *v71 != &v96 )
    {
      if ( v72 == ++v71 )
        goto LABEL_116;
    }
  }
LABEL_104:
  v12 = 0;
  (*(void (__fastcall **)(_QWORD, void **))(**(_QWORD **)(a1 + 976) + 24LL))(*(_QWORD *)(a1 + 976), &v96);
  v73 = v99;
  v96 = &unk_4A388F0;
  if ( *(_BYTE *)(v99 + 36) )
  {
    v74 = *(_QWORD ***)(v99 + 16);
    v75 = &v74[*(unsigned int *)(v99 + 28)];
    v76 = v74;
    if ( v74 != v75 )
    {
      while ( *v76 != &v96 )
      {
        if ( v75 == ++v76 )
          goto LABEL_110;
      }
      v77 = (unsigned int)(*(_DWORD *)(v99 + 28) - 1);
      *(_DWORD *)(v99 + 28) = v77;
      *v76 = v74[v77];
      ++*(_QWORD *)(v73 + 8);
    }
  }
  else
  {
    v78 = sub_C8CA60(v99 + 8, (__int64)&v96);
    if ( v78 )
    {
      *v78 = -2;
      ++*(_DWORD *)(v73 + 32);
      ++*(_QWORD *)(v73 + 8);
    }
  }
LABEL_110:
  if ( !v117 )
    _libc_free((unsigned __int64)v114);
  if ( !v111 )
    _libc_free((unsigned __int64)v108);
LABEL_64:
  if ( v88 != (unsigned __int16 *)&v90 )
    _libc_free((unsigned __int64)v88);
  if ( v85 != (unsigned int *)v87 )
    _libc_free((unsigned __int64)v85);
  return v12;
}
