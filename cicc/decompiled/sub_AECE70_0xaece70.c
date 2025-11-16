// Function: sub_AECE70
// Address: 0xaece70
//
__int64 __fastcall sub_AECE70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rdi
  unsigned __int8 v6; // cl
  __int64 v7; // rcx
  __int64 v8; // r9
  unsigned int v9; // r8d
  __int64 v10; // r11
  unsigned __int8 v11; // cl
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int8 v14; // cl
  __int64 v15; // rsi
  __int64 v16; // r12
  __int64 v17; // rcx
  __int64 v18; // rdi
  unsigned int v19; // r9d
  __int64 *v20; // rsi
  __int64 v21; // r10
  unsigned __int8 v22; // cl
  __int64 v23; // rsi
  __int64 v24; // r15
  __int64 v25; // rcx
  __int64 v26; // r9
  unsigned int v27; // edi
  __int64 *v28; // rsi
  __int64 v29; // r11
  unsigned __int8 v30; // cl
  __int64 v31; // rsi
  __int64 v32; // r14
  __int64 v33; // rsi
  __int64 v34; // r9
  unsigned int v35; // r11d
  __int64 *v36; // rdi
  __int64 v37; // r13
  __int64 v38; // rsi
  int v39; // r13d
  __int64 v40; // rdx
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // r11
  __int64 v44; // rax
  _QWORD *v45; // rdi
  int v46; // ecx
  int v47; // eax
  int v48; // edx
  int v49; // eax
  __int64 v50; // r12
  unsigned __int8 v51; // dl
  __int64 v52; // rax
  const void *v53; // rdi
  size_t v54; // rdx
  size_t v55; // r8
  __int64 v56; // rsi
  __int64 v57; // rcx
  unsigned int v58; // eax
  __int64 *v59; // rdx
  __int64 v60; // r9
  __int64 v62; // rsi
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // edx
  unsigned __int8 v67; // dl
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned int v72; // eax
  __int64 *v73; // rdx
  __int64 v74; // rdi
  int v75; // r10d
  __int64 *v76; // r9
  int v77; // eax
  int v78; // eax
  __int64 v79; // rdx
  int v80; // esi
  int v81; // r11d
  int v82; // edi
  int v83; // ecx
  int v84; // esi
  int v85; // r8d
  unsigned int v86; // r12d
  int v87; // r11d
  __int64 v88; // [rsp+0h] [rbp-130h]
  __int64 v89; // [rsp+8h] [rbp-128h]
  int v90; // [rsp+8h] [rbp-128h]
  __int64 v91; // [rsp+18h] [rbp-118h]
  int v92; // [rsp+20h] [rbp-110h]
  int v93; // [rsp+24h] [rbp-10Ch]
  int v94; // [rsp+28h] [rbp-108h]
  int v95; // [rsp+2Ch] [rbp-104h]
  int v96; // [rsp+30h] [rbp-100h]
  int v97; // [rsp+34h] [rbp-FCh]
  __int64 v98; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v99; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v100; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v101; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v102; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v103; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v104; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v105; // [rsp+70h] [rbp-C0h] BYREF
  __int64 *v106; // [rsp+78h] [rbp-B8h] BYREF
  const char *v107; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v108; // [rsp+88h] [rbp-A8h]
  __int64 v109; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v110; // [rsp+98h] [rbp-98h] BYREF
  __int64 *v111[16]; // [rsp+B0h] [rbp-80h] BYREF

  v2 = a2;
  v3 = a2 - 16;
  v5 = a2;
  v98 = a2;
  if ( *(_BYTE *)a2 == 16
    || ((v6 = *(_BYTE *)(a2 - 16), (v6 & 2) != 0) ? (a2 = *(_QWORD *)(a2 - 32)) : (a2 = v3 - 8LL * ((v6 >> 2) & 0xF)),
        (v5 = *(_QWORD *)a2) != 0) )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v7 )
    {
      v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      a2 = v8 + 16LL * v9;
      v10 = *(_QWORD *)a2;
      if ( *(_QWORD *)a2 == v5 )
      {
LABEL_7:
        if ( a2 != v8 + 16 * v7 )
          v5 = *(_QWORD *)(a2 + 8);
      }
      else
      {
        a2 = 1;
        while ( v10 != -4096 )
        {
          v86 = a2 + 1;
          v9 = (v7 - 1) & (a2 + v9);
          a2 = v8 + 16LL * v9;
          v10 = *(_QWORD *)a2;
          if ( *(_QWORD *)a2 == v5 )
            goto LABEL_7;
          a2 = v86;
        }
      }
    }
  }
  v99 = v5;
  v11 = *(_BYTE *)(v2 - 16);
  if ( (v11 & 2) != 0 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v2 - 32) + 16LL);
    if ( !v12 )
      goto LABEL_68;
  }
  else
  {
    a2 = v3 - 8LL * ((v11 >> 2) & 0xF);
    v12 = *(_QWORD *)(a2 + 16);
    if ( !v12 )
    {
LABEL_73:
      v62 = v3 - 8LL * ((*(_BYTE *)(v2 - 16) >> 2) & 0xF);
      goto LABEL_69;
    }
  }
  sub_B91420(v12, a2);
  if ( v13 )
  {
    v108 = 0;
    v107 = byte_3F871B3;
    v2 = v98;
    v3 = v98 - 16;
    goto LABEL_13;
  }
  v2 = v98;
  v3 = v98 - 16;
  if ( (*(_BYTE *)(v98 - 16) & 2) == 0 )
    goto LABEL_73;
LABEL_68:
  v62 = *(_QWORD *)(v2 - 32);
LABEL_69:
  v63 = *(_QWORD *)(v62 + 24);
  if ( v63 )
  {
    v63 = sub_B91420(v63, v62);
    v2 = v98;
    v65 = v64;
    v3 = v98 - 16;
  }
  else
  {
    v65 = 0;
  }
  v107 = (const char *)v63;
  v108 = v65;
LABEL_13:
  v100 = 0;
  v14 = *(_BYTE *)(v2 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(v2 - 32);
  else
    v15 = v3 - 8LL * ((v14 >> 2) & 0xF);
  v16 = *(_QWORD *)(v15 + 32);
  if ( v16 )
  {
    v17 = *(unsigned int *)(a1 + 24);
    v18 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v17 )
    {
      v19 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v16 == *v20 )
      {
LABEL_18:
        if ( v20 != (__int64 *)(v18 + 16 * v17) )
          v16 = v20[1];
      }
      else
      {
        v80 = 1;
        while ( v21 != -4096 )
        {
          v81 = v80 + 1;
          v19 = (v17 - 1) & (v80 + v19);
          v20 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v20;
          if ( v16 == *v20 )
            goto LABEL_18;
          v80 = v81;
        }
      }
    }
  }
  v101 = v16;
  v22 = *(_BYTE *)(v2 - 16);
  if ( (v22 & 2) != 0 )
  {
    if ( *(_DWORD *)(v2 - 24) <= 8u )
      goto LABEL_61;
    v23 = *(_QWORD *)(v2 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v2 - 16) >> 6) & 0xFu) <= 8 )
      goto LABEL_61;
    v23 = v3 - 8LL * ((v22 >> 2) & 0xF);
  }
  v24 = *(_QWORD *)(v23 + 64);
  if ( v24 )
  {
    v25 = *(unsigned int *)(a1 + 24);
    v26 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v25 )
    {
      v27 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v24 == *v28 )
      {
LABEL_26:
        if ( v28 != (__int64 *)(v26 + 16 * v25) )
          v24 = v28[1];
      }
      else
      {
        v84 = 1;
        while ( v29 != -4096 )
        {
          v85 = v84 + 1;
          v27 = (v25 - 1) & (v84 + v27);
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v24 == *v28 )
            goto LABEL_26;
          v84 = v85;
        }
      }
    }
    v102 = v24;
    v30 = *(_BYTE *)(v2 - 16);
    if ( (v30 & 2) != 0 )
      goto LABEL_29;
LABEL_62:
    v31 = v3 - 8LL * ((v30 >> 2) & 0xF);
    goto LABEL_30;
  }
LABEL_61:
  v24 = 0;
  v102 = 0;
  v30 = *(_BYTE *)(v2 - 16);
  if ( (v30 & 2) == 0 )
    goto LABEL_62;
LABEL_29:
  v31 = *(_QWORD *)(v2 - 32);
LABEL_30:
  v32 = *(_QWORD *)(v31 + 40);
  if ( v32 )
  {
    v33 = *(unsigned int *)(a1 + 24);
    v34 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v33 )
    {
      v35 = (v33 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v36 = (__int64 *)(v34 + 16LL * v35);
      v37 = *v36;
      if ( v32 == *v36 )
      {
LABEL_33:
        if ( v36 != (__int64 *)(v34 + 16 * v33) )
          v32 = v36[1];
      }
      else
      {
        v82 = 1;
        while ( v37 != -4096 )
        {
          v83 = v82 + 1;
          v35 = (v33 - 1) & (v82 + v35);
          v36 = (__int64 *)(v34 + 16LL * v35);
          v37 = *v36;
          if ( v32 == *v36 )
            goto LABEL_33;
          v82 = v83;
        }
      }
    }
  }
  v103 = v32;
  v111[0] = &v98;
  v111[1] = &v99;
  v111[2] = (__int64 *)&v107;
  v111[3] = &v101;
  v111[4] = &v102;
  v111[5] = &v103;
  v111[6] = &v105;
  v111[7] = &v100;
  v104 = 0;
  v105 = 0;
  v111[8] = &v104;
  if ( (*(_BYTE *)(v2 + 1) & 0x7F) == 1 )
    return sub_AE5EC0(v111);
  v38 = *(unsigned __int8 *)(v2 - 16);
  v39 = v99;
  v92 = *(_DWORD *)(v2 + 36);
  v95 = *(_DWORD *)(v2 + 24);
  v93 = *(_DWORD *)(v2 + 32);
  v97 = *(_DWORD *)(v2 + 16);
  v94 = *(_DWORD *)(v2 + 28);
  v96 = *(_DWORD *)(v2 + 20);
  if ( (v38 & 2) != 0 )
  {
    v40 = *(_QWORD *)(v2 - 32);
  }
  else
  {
    v38 = 8LL * (((unsigned __int8)v38 >> 2) & 0xF);
    v40 = v3 - v38;
  }
  v41 = *(_QWORD *)(v40 + 16);
  if ( v41 )
  {
    v41 = sub_B91420(*(_QWORD *)(v40 + 16), v38);
    v43 = v42;
    v91 = v99;
    v2 = v98;
  }
  else
  {
    v91 = v99;
    v43 = 0;
  }
  v44 = *(_QWORD *)(v2 + 8);
  v45 = (_QWORD *)(v44 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v44 & 4) != 0 )
    v45 = (_QWORD *)*v45;
  v46 = 0;
  if ( v108 )
  {
    v88 = v43;
    v89 = v41;
    v47 = sub_B9B140(v45, v107, v108);
    v43 = v88;
    v41 = v89;
    v46 = v47;
  }
  v48 = 0;
  if ( v43 )
  {
    v90 = v46;
    v49 = sub_B9B140(v45, v41, v43);
    v46 = v90;
    v48 = v49;
  }
  v50 = sub_B07EA0((_DWORD)v45, v91, v48, v46, v39, v97, v16, v96, v24, v95, v94, v93, v92, v32, 0, 0, 0, 0, 0, 0, 0, 1);
  v51 = *(_BYTE *)(v98 - 16);
  if ( (v51 & 2) != 0 )
    v52 = *(_QWORD *)(v98 - 32);
  else
    v52 = v98 - 16 - 8LL * ((v51 >> 2) & 0xF);
  v53 = *(const void **)(v52 + 24);
  if ( v53 )
  {
    v53 = (const void *)sub_B91420(v53, v91);
    v55 = v54;
  }
  else
  {
    v55 = 0;
  }
  v56 = *(unsigned int *)(a1 + 64);
  v57 = *(_QWORD *)(a1 + 48);
  if ( (_DWORD)v56 )
  {
    v58 = (v56 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
    v59 = (__int64 *)(v57 + 24LL * v58);
    v60 = *v59;
    if ( *v59 == v50 )
    {
LABEL_52:
      if ( v59 != (__int64 *)(v57 + 24LL * (unsigned int)v56) )
      {
        if ( v55 == v59[2] && (!v55 || !memcmp((const void *)v59[1], v53, v55)) )
          return v50;
        return sub_AE5EC0(v111);
      }
    }
    else
    {
      v66 = 1;
      while ( v60 != -4096 )
      {
        v87 = v66 + 1;
        v58 = (v56 - 1) & (v66 + v58);
        v59 = (__int64 *)(v57 + 24LL * v58);
        v60 = *v59;
        if ( v50 == *v59 )
          goto LABEL_52;
        v66 = v87;
      }
    }
  }
  v67 = *(_BYTE *)(v98 - 16);
  if ( (v67 & 2) != 0 )
    v68 = *(_QWORD *)(v98 - 32);
  else
    v68 = v98 - 16 - 8LL * ((v67 >> 2) & 0xF);
  v69 = *(_QWORD *)(v68 + 24);
  if ( v69 )
  {
    v70 = sub_B91420(v69, v56);
    v57 = *(_QWORD *)(a1 + 48);
    LODWORD(v56) = *(_DWORD *)(a1 + 64);
    v69 = v70;
  }
  else
  {
    v71 = 0;
  }
  v109 = v50;
  v110.m128i_i64[0] = v69;
  v110.m128i_i64[1] = v71;
  if ( !(_DWORD)v56 )
  {
    ++*(_QWORD *)(a1 + 40);
    v106 = 0;
LABEL_115:
    LODWORD(v56) = 2 * v56;
    goto LABEL_116;
  }
  v72 = (v56 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
  v73 = (__int64 *)(v57 + 24LL * v72);
  v74 = *v73;
  if ( v50 == *v73 )
    return v50;
  v75 = 1;
  v76 = 0;
  while ( v74 != -4096 )
  {
    if ( v74 == -8192 && !v76 )
      v76 = v73;
    v72 = (v56 - 1) & (v75 + v72);
    v73 = (__int64 *)(v57 + 24LL * v72);
    v74 = *v73;
    if ( v50 == *v73 )
      return v50;
    ++v75;
  }
  v77 = *(_DWORD *)(a1 + 56);
  if ( !v76 )
    v76 = v73;
  ++*(_QWORD *)(a1 + 40);
  v78 = v77 + 1;
  v106 = v76;
  if ( 4 * v78 >= (unsigned int)(3 * v56) )
    goto LABEL_115;
  v79 = v50;
  if ( (int)v56 - (v78 + *(_DWORD *)(a1 + 60)) <= (unsigned int)v56 >> 3 )
  {
LABEL_116:
    sub_AECC70(a1 + 40, v56);
    sub_AEAA10(a1 + 40, &v109, &v106);
    v79 = v109;
    v76 = v106;
    v78 = *(_DWORD *)(a1 + 56) + 1;
  }
  *(_DWORD *)(a1 + 56) = v78;
  if ( *v76 != -4096 )
    --*(_DWORD *)(a1 + 60);
  *v76 = v79;
  *(__m128i *)(v76 + 1) = _mm_loadu_si128(&v110);
  return v50;
}
