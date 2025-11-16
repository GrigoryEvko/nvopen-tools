// Function: sub_124E8C0
// Address: 0x124e8c0
//
_QWORD *__fastcall sub_124E8C0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r13
  _QWORD *v16; // rdx
  _QWORD *v17; // rsi
  _QWORD *result; // rax
  char v19; // cl
  __int64 v20; // rdi
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r10
  __int64 v24; // r13
  unsigned int v25; // eax
  __int64 v26; // r14
  __int32 v27; // r15d
  __int64 v28; // r10
  char v29; // al
  __int64 v30; // r10
  bool v31; // zf
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 v34; // r9
  unsigned int v35; // ecx
  _QWORD *v36; // rdx
  __int64 v37; // rax
  __m128i *v38; // rax
  __m128i *v39; // rsi
  const __m128i **v40; // rdi
  __int64 v41; // rax
  int v42; // edx
  __int64 v43; // rcx
  int v44; // edx
  unsigned int v45; // esi
  __int64 *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  unsigned int v49; // esi
  __int64 v50; // r9
  unsigned int v51; // ecx
  __int64 v52; // rdx
  __int64 v53; // rax
  _QWORD *v54; // rsi
  __int64 v55; // rax
  void *v56; // rax
  int v57; // eax
  int v58; // ecx
  __int64 v59; // r8
  unsigned int v60; // eax
  int v61; // edx
  _QWORD *v62; // rdi
  __int64 v63; // rsi
  void *v64; // rax
  __int64 *v65; // rdx
  int v66; // r11d
  int v67; // eax
  int v68; // eax
  int v69; // eax
  __int64 v70; // rsi
  _QWORD *v71; // r8
  unsigned int v72; // r12d
  int v73; // r9d
  __int64 v74; // rcx
  int v75; // eax
  int v76; // r8d
  int v77; // r11d
  int v78; // eax
  int v79; // eax
  int v80; // eax
  __int64 v81; // rsi
  unsigned int v82; // r12d
  int v83; // r9d
  __int64 v84; // rcx
  int v85; // eax
  int v86; // ecx
  __int64 v87; // r8
  unsigned int v88; // eax
  __int64 v89; // rsi
  int v90; // r10d
  _QWORD *v91; // r9
  int v92; // r10d
  char v93; // al
  __int64 v94; // [rsp+8h] [rbp-D8h]
  char v95; // [rsp+17h] [rbp-C9h]
  _QWORD *v96; // [rsp+18h] [rbp-C8h]
  char v97; // [rsp+18h] [rbp-C8h]
  __int64 v98; // [rsp+20h] [rbp-C0h]
  __int64 v99; // [rsp+20h] [rbp-C0h]
  __int64 v100; // [rsp+20h] [rbp-C0h]
  __int64 v101; // [rsp+20h] [rbp-C0h]
  __int64 v102; // [rsp+20h] [rbp-C0h]
  __int64 v103; // [rsp+20h] [rbp-C0h]
  __int64 v105; // [rsp+30h] [rbp-B0h]
  __int64 v106; // [rsp+38h] [rbp-A8h]
  __int64 v107; // [rsp+40h] [rbp-A0h]
  __int64 v108; // [rsp+48h] [rbp-98h]
  _QWORD v109[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v110; // [rsp+70h] [rbp-70h]
  __m128i v111; // [rsp+80h] [rbp-60h] BYREF
  __m128i v112; // [rsp+90h] [rbp-50h] BYREF
  __int16 v113; // [rsp+A0h] [rbp-40h]

  v12 = *(_DWORD *)((*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2[1] + 64LL))(
                      a2[1],
                      *(unsigned int *)(a4 + 12))
                  + 16);
  v108 = *(_QWORD *)(a3 + 8);
  v107 = a9;
  v13 = sub_E5C2C0((__int64)a2, a3);
  v14 = *a2;
  v106 = *(unsigned int *)(a4 + 8) + v13;
  v105 = *(_QWORD *)(*a2 + 2368);
  if ( !a8 )
  {
    v23 = a7;
    v97 = v12 & 1;
    if ( !a7 )
      goto LABEL_11;
    goto LABEL_31;
  }
  v15 = *(_QWORD *)(a8 + 16);
  v16 = *(_QWORD **)v15;
  if ( !*(_QWORD *)v15 )
  {
    v19 = *(_BYTE *)(v15 + 8);
    if ( (*(_BYTE *)(v15 + 9) & 0x70) == 0x20 && v19 >= 0 )
    {
      v20 = *(_QWORD *)(v15 + 24);
      v96 = *(_QWORD **)v15;
      *(_BYTE *)(v15 + 8) = v19 | 8;
      v98 = v14;
      v21 = sub_E807D0(v20);
      v14 = v98;
      v16 = v96;
      *(_QWORD *)v15 = v21;
      if ( v21 )
      {
        if ( v108 != v21[1] )
          goto LABEL_4;
        goto LABEL_10;
      }
      v19 = *(_BYTE *)(v15 + 8);
    }
    v53 = 0;
    if ( (v19 & 1) != 0 )
    {
      v65 = *(__int64 **)(v15 - 8);
      v53 = *v65;
      v16 = v65 + 3;
    }
    v109[3] = v53;
    v54 = *(_QWORD **)(a4 + 16);
    v109[2] = v16;
    v110 = 1283;
    v111.m128i_i64[0] = (__int64)v109;
    v113 = 770;
    v109[0] = "symbol '";
    v112.m128i_i64[0] = (__int64)"' can not be undefined in a subtraction expression";
    return (_QWORD *)sub_E66880(v14, v54, (__int64)&v111);
  }
  if ( v108 != v16[1] )
  {
LABEL_4:
    v17 = *(_QWORD **)(a4 + 16);
    v113 = 259;
    v111.m128i_i64[0] = (__int64)"Cannot represent a difference across sections";
    return (_QWORD *)sub_E66880(v14, v17, (__int64)&v111);
  }
LABEL_10:
  v99 = v14;
  v22 = sub_E5C4C0((__int64)a2, v15);
  v23 = a7;
  v97 = 1;
  v14 = v99;
  v107 = v106 + v107 - v22;
  if ( !a7 )
  {
LABEL_11:
    v95 = 0;
    v24 = 0;
    goto LABEL_12;
  }
LABEL_31:
  v24 = *(_QWORD *)(v23 + 16);
  if ( !v24 )
  {
    v95 = 0;
    v23 = 0;
    goto LABEL_12;
  }
  if ( (*(_BYTE *)(v24 + 9) & 0x70) != 0x20 )
  {
    v23 = *(_QWORD *)v24;
    v95 = 0;
    if ( !*(_QWORD *)v24 )
      goto LABEL_12;
    goto LABEL_34;
  }
  v55 = *(_QWORD *)(v24 + 24);
  *(_BYTE *)(v24 + 8) |= 8u;
  if ( *(_BYTE *)v55 != 2 || *(_WORD *)(v55 + 1) != 29 )
  {
    v23 = *(_QWORD *)v24;
    v95 = 0;
    if ( !*(_QWORD *)v24 )
      goto LABEL_60;
LABEL_34:
    if ( (_UNKNOWN *)v23 == off_4C5D170 )
      v23 = 0;
    else
      v23 = *(_QWORD *)(v23 + 8);
    goto LABEL_12;
  }
  v24 = *(_QWORD *)(v55 + 16);
  if ( !v24 )
  {
    v95 = 1;
    v23 = 0;
    goto LABEL_12;
  }
  v23 = *(_QWORD *)v24;
  if ( *(_QWORD *)v24 )
  {
    v95 = 1;
    goto LABEL_34;
  }
  v95 = 1;
  if ( (*(_BYTE *)(v24 + 9) & 0x70) != 0x20 )
    goto LABEL_12;
LABEL_60:
  v23 = 0;
  if ( *(char *)(v24 + 8) >= 0 )
  {
    *(_BYTE *)(v24 + 8) |= 8u;
    v103 = v14;
    v56 = sub_E807D0(*(_QWORD *)(v24 + 24));
    v14 = v103;
    *(_QWORD *)v24 = v56;
    v23 = (__int64)v56;
    if ( v56 )
      goto LABEL_34;
  }
LABEL_12:
  v94 = v23;
  v100 = v14;
  result = (_QWORD *)sub_124CA80(a1, v14, *(_QWORD **)(a4 + 16), v108, v23);
  if ( !(_BYTE)result )
    return result;
  v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, _QWORD))(**(_QWORD **)(a1 + 112) + 24LL))(
          *(_QWORD *)(a1 + 112),
          v100,
          &a7,
          a4,
          v97 & 1);
  v26 = *(_QWORD *)(a3 + 8);
  v27 = v25;
  if ( !(unsigned __int8)sub_124C870(a1, (__int64)a2, &a7, v24, v107, v25) && *(_DWORD *)(v26 + 148) != 1879002121 )
  {
    v28 = v94;
    if ( v24 )
    {
      if ( *(_QWORD *)v24
        || (*(_BYTE *)(v24 + 9) & 0x70) == 0x20
        && *(char *)(v24 + 8) >= 0
        && (*(_BYTE *)(v24 + 8) |= 8u, v64 = sub_E807D0(*(_QWORD *)(v24 + 24)), v28 = v94, (*(_QWORD *)v24 = v64) != 0) )
      {
        v101 = v28;
        v107 += sub_E5C4C0((__int64)a2, v24);
        v28 = v101;
      }
      v102 = v28;
      v29 = sub_124CB30(a1, v105, v108);
      v30 = v102;
      v31 = v29 == 0;
      v32 = 0;
      if ( v31 )
        v32 = v107;
      *a5 = v32;
    }
    else
    {
      v93 = sub_124CB30(a1, v105, v108);
      v30 = v94;
      if ( !v93 )
        v24 = v107;
      *a5 = v24;
    }
    if ( v30 )
    {
      v30 = *(_QWORD *)(v30 + 16);
      if ( v30 )
        *(_BYTE *)(v30 + 9) |= 8u;
    }
    v33 = *(_DWORD *)(a1 + 160);
    v111.m128i_i64[1] = v30;
    v112.m128i_i32[0] = v27;
    v111.m128i_i64[0] = v106;
    v112.m128i_i64[1] = v107;
    if ( v33 )
    {
      v34 = *(_QWORD *)(a1 + 144);
      v35 = (v33 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
      v36 = (_QWORD *)(v34 + 32LL * v35);
      v37 = *v36;
      if ( v108 == *v36 )
      {
LABEL_26:
        v38 = (__m128i *)v36[2];
        v39 = (__m128i *)v36[3];
        v40 = (const __m128i **)(v36 + 1);
        if ( v38 != v39 )
        {
          if ( v38 )
          {
            *v38 = _mm_loadu_si128(&v111);
            v38[1] = _mm_loadu_si128(&v112);
            v38 = (__m128i *)v36[2];
          }
          result = v38[2].m128i_i64;
          v36[2] = result;
          return result;
        }
        return (_QWORD *)sub_124CDA0(v40, v39, &v111);
      }
      v77 = 1;
      v62 = 0;
      while ( v37 != -4096 )
      {
        if ( v37 == -8192 && !v62 )
          v62 = v36;
        v35 = (v33 - 1) & (v77 + v35);
        v36 = (_QWORD *)(v34 + 32LL * v35);
        v37 = *v36;
        if ( v108 == *v36 )
          goto LABEL_26;
        ++v77;
      }
      v78 = *(_DWORD *)(a1 + 152);
      if ( !v62 )
        v62 = v36;
      ++*(_QWORD *)(a1 + 136);
      v61 = v78 + 1;
      if ( 4 * (v78 + 1) < 3 * v33 )
      {
        if ( v33 - *(_DWORD *)(a1 + 156) - v61 > v33 >> 3 )
          goto LABEL_70;
        sub_124E280(a1 + 136, v33);
        v79 = *(_DWORD *)(a1 + 160);
        if ( v79 )
        {
          v80 = v79 - 1;
          v81 = *(_QWORD *)(a1 + 144);
          v71 = 0;
          v82 = v80 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
          v83 = 1;
          v61 = *(_DWORD *)(a1 + 152) + 1;
          v62 = (_QWORD *)(v81 + 32LL * v82);
          v84 = *v62;
          if ( v108 != *v62 )
          {
            while ( v84 != -4096 )
            {
              if ( !v71 && v84 == -8192 )
                v71 = v62;
              v82 = v80 & (v83 + v82);
              v62 = (_QWORD *)(v81 + 32LL * v82);
              v84 = *v62;
              if ( v108 == *v62 )
                goto LABEL_70;
              ++v83;
            }
LABEL_89:
            if ( v71 )
              v62 = v71;
            goto LABEL_70;
          }
          goto LABEL_70;
        }
LABEL_150:
        ++*(_DWORD *)(a1 + 152);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 136);
    }
    sub_124E280(a1 + 136, 2 * v33);
    v85 = *(_DWORD *)(a1 + 160);
    if ( !v85 )
      goto LABEL_150;
    v86 = v85 - 1;
    v87 = *(_QWORD *)(a1 + 144);
    v88 = (v85 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
    v61 = *(_DWORD *)(a1 + 152) + 1;
    v62 = (_QWORD *)(v87 + 32LL * v88);
    v89 = *v62;
    if ( v108 == *v62 )
      goto LABEL_70;
    v90 = 1;
    v91 = 0;
    while ( v89 != -4096 )
    {
      if ( !v91 && v89 == -8192 )
        v91 = v62;
      v88 = v86 & (v90 + v88);
      v62 = (_QWORD *)(v87 + 32LL * v88);
      v89 = *v62;
      if ( v108 == *v62 )
        goto LABEL_70;
      ++v90;
    }
LABEL_115:
    if ( v91 )
      v62 = v91;
    goto LABEL_70;
  }
  v31 = (unsigned __int8)sub_124CB30(a1, v105, v108) == 0;
  v41 = 0;
  if ( v31 )
    v41 = v107;
  *a5 = v41;
  if ( v24 )
  {
    v42 = *(_DWORD *)(a1 + 192);
    v43 = *(_QWORD *)(a1 + 176);
    if ( v42 )
    {
      v44 = v42 - 1;
      v45 = v44 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v46 = (__int64 *)(v43 + 16LL * v45);
      v47 = *v46;
      if ( v24 == *v46 )
      {
LABEL_41:
        v48 = v46[1];
        if ( v48 )
          v24 = v48;
      }
      else
      {
        v75 = 1;
        while ( v47 != -4096 )
        {
          v76 = v75 + 1;
          v45 = v44 & (v75 + v45);
          v46 = (__int64 *)(v43 + 16LL * v45);
          v47 = *v46;
          if ( v24 == *v46 )
            goto LABEL_41;
          v75 = v76;
        }
      }
    }
    if ( v95 )
      sub_EA16C0(v24);
    else
      *(_BYTE *)(v24 + 9) |= 8u;
  }
  v49 = *(_DWORD *)(a1 + 160);
  v111.m128i_i64[1] = v24;
  v112.m128i_i32[0] = v27;
  v111.m128i_i64[0] = v106;
  v112.m128i_i64[1] = v107;
  if ( !v49 )
  {
    ++*(_QWORD *)(a1 + 136);
    goto LABEL_68;
  }
  v50 = *(_QWORD *)(a1 + 144);
  v51 = (v49 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
  result = (_QWORD *)(v50 + 32LL * v51);
  v52 = *result;
  if ( v108 != *result )
  {
    v66 = 1;
    v62 = 0;
    while ( v52 != -4096 )
    {
      if ( !v62 && v52 == -8192 )
        v62 = result;
      v51 = (v49 - 1) & (v66 + v51);
      result = (_QWORD *)(v50 + 32LL * v51);
      v52 = *result;
      if ( v108 == *result )
        goto LABEL_47;
      ++v66;
    }
    if ( !v62 )
      v62 = result;
    v67 = *(_DWORD *)(a1 + 152);
    ++*(_QWORD *)(a1 + 136);
    v61 = v67 + 1;
    if ( 4 * (v67 + 1) < 3 * v49 )
    {
      if ( v49 - *(_DWORD *)(a1 + 156) - v61 > v49 >> 3 )
        goto LABEL_70;
      sub_124E280(a1 + 136, v49);
      v68 = *(_DWORD *)(a1 + 160);
      if ( v68 )
      {
        v69 = v68 - 1;
        v70 = *(_QWORD *)(a1 + 144);
        v71 = 0;
        v72 = v69 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
        v73 = 1;
        v61 = *(_DWORD *)(a1 + 152) + 1;
        v62 = (_QWORD *)(v70 + 32LL * v72);
        v74 = *v62;
        if ( v108 != *v62 )
        {
          while ( v74 != -4096 )
          {
            if ( !v71 && v74 == -8192 )
              v71 = v62;
            v72 = v69 & (v73 + v72);
            v62 = (_QWORD *)(v70 + 32LL * v72);
            v74 = *v62;
            if ( v108 == *v62 )
              goto LABEL_70;
            ++v73;
          }
          goto LABEL_89;
        }
LABEL_70:
        *(_DWORD *)(a1 + 152) = v61;
        if ( *v62 != -4096 )
          --*(_DWORD *)(a1 + 156);
        v40 = (const __m128i **)(v62 + 1);
        *v40 = 0;
        v39 = 0;
        v40[1] = 0;
        *(v40 - 1) = (const __m128i *)v108;
        v40[2] = 0;
        return (_QWORD *)sub_124CDA0(v40, v39, &v111);
      }
      goto LABEL_149;
    }
LABEL_68:
    sub_124E280(a1 + 136, 2 * v49);
    v57 = *(_DWORD *)(a1 + 160);
    if ( v57 )
    {
      v58 = v57 - 1;
      v59 = *(_QWORD *)(a1 + 144);
      v60 = (v57 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
      v61 = *(_DWORD *)(a1 + 152) + 1;
      v62 = (_QWORD *)(v59 + 32LL * v60);
      v63 = *v62;
      if ( v108 == *v62 )
        goto LABEL_70;
      v92 = 1;
      v91 = 0;
      while ( v63 != -4096 )
      {
        if ( v63 == -8192 && !v91 )
          v91 = v62;
        v60 = v58 & (v92 + v60);
        v62 = (_QWORD *)(v59 + 32LL * v60);
        v63 = *v62;
        if ( v108 == *v62 )
          goto LABEL_70;
        ++v92;
      }
      goto LABEL_115;
    }
LABEL_149:
    ++*(_DWORD *)(a1 + 152);
    BUG();
  }
LABEL_47:
  v39 = (__m128i *)result[2];
  v40 = (const __m128i **)(result + 1);
  if ( (__m128i *)result[3] == v39 )
    return (_QWORD *)sub_124CDA0(v40, v39, &v111);
  if ( v39 )
  {
    *v39 = _mm_loadu_si128(&v111);
    v39[1] = _mm_loadu_si128(&v112);
    v39 = (__m128i *)result[2];
  }
  result[2] = v39 + 2;
  return result;
}
