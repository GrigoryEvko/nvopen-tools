// Function: sub_308E990
// Address: 0x308e990
//
void __fastcall sub_308E990(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // eax
  __int64 v7; // rdx
  unsigned int v8; // r15d
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // eax
  int v15; // edi
  __m128i *v16; // r13
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // r10
  unsigned int v21; // ebx
  int v22; // r12d
  __int64 v23; // r8
  __int64 v24; // rdx
  __int16 ***v25; // rdx
  int v26; // ecx
  __m128i *v27; // r13
  __m128i *v28; // rcx
  __int64 v29; // rdx
  __m128i *v30; // rdx
  __m128i v31; // xmm1
  __int64 v32; // rax
  int v33; // ecx
  __int64 v34; // r8
  unsigned int v35; // esi
  int *v36; // rdi
  int v37; // r9d
  char *v38; // r13
  __int64 v39; // r12
  __int64 v40; // r10
  unsigned int v41; // esi
  __int64 v42; // r8
  unsigned int v43; // edi
  int *v44; // rax
  int v45; // ecx
  unsigned int ***v46; // r14
  __m128i *v47; // rbx
  __m128i *v48; // r12
  unsigned int **v49; // rdi
  int v50; // eax
  __int64 v51; // r9
  __int32 v52; // r10d
  __int64 v53; // rcx
  __int32 v54; // edx
  __int64 v55; // r8
  unsigned __int64 v56; // r13
  __int64 v57; // rcx
  __m128i *v58; // rsi
  __m128i *v59; // rdx
  __m128i *v60; // rcx
  int v61; // r11d
  int v62; // edx
  int v63; // r8d
  int v64; // r14d
  int *v65; // rdx
  int v66; // eax
  int v67; // ecx
  unsigned int **v68; // rax
  __m128i *v69; // r8
  unsigned __int64 v70; // rdx
  __m128i *v71; // rax
  bool v72; // zf
  int v73; // edi
  int v74; // r11d
  int v75; // r9d
  int v76; // r9d
  __int64 v77; // r10
  unsigned int v78; // eax
  int v79; // r8d
  int v80; // edi
  int *v81; // rsi
  int v82; // edi
  int v83; // edi
  __int64 v84; // r9
  int v85; // esi
  unsigned int v86; // ebx
  int *v87; // rax
  int v88; // r8d
  __int64 v89; // r8
  __int64 v90; // [rsp+0h] [rbp-290h]
  __int64 v91; // [rsp+8h] [rbp-288h]
  unsigned int v92; // [rsp+10h] [rbp-280h]
  __int64 v93; // [rsp+10h] [rbp-280h]
  int v94; // [rsp+1Ch] [rbp-274h]
  unsigned int v95; // [rsp+20h] [rbp-270h]
  __m128i v96; // [rsp+30h] [rbp-260h] BYREF
  __m128i v97; // [rsp+40h] [rbp-250h] BYREF
  __m128i *v98; // [rsp+50h] [rbp-240h] BYREF
  __int64 v99; // [rsp+58h] [rbp-238h]
  _BYTE v100[560]; // [rsp+60h] [rbp-230h] BYREF

  v98 = (__m128i *)v100;
  v99 = 0x1000000000LL;
  v6 = sub_3089AC0(a1, a2);
  if ( !v6 )
    goto LABEL_10;
  v7 = *(_QWORD *)(a2 + 32);
  v8 = v6;
  v9 = *(_QWORD *)(v7 + 40LL * (v6 + 2) + 24);
  if ( (_DWORD)v9 != 5 )
  {
    if ( (_DWORD)v9 )
      goto LABEL_10;
  }
  v10 = v7 + 40LL * (v8 + 6);
  if ( *(_BYTE *)v10 )
  {
    if ( *(_BYTE *)v10 != 5 )
      goto LABEL_10;
    v94 = *(_DWORD *)(v10 + 24);
    if ( v94 == -1 )
      goto LABEL_10;
  }
  else
  {
    v32 = *(unsigned int *)(a3 + 24);
    v33 = *(_DWORD *)(v10 + 8);
    v34 = *(_QWORD *)(a3 + 8);
    if ( !(_DWORD)v32 )
      goto LABEL_10;
    v35 = (v32 - 1) & (37 * v33);
    v36 = (int *)(v34 + 8LL * v35);
    v37 = *v36;
    if ( v33 != *v36 )
    {
      v73 = 1;
      while ( v37 != -1 )
      {
        v74 = v73 + 1;
        v35 = (v32 - 1) & (v73 + v35);
        v36 = (int *)(v34 + 8LL * v35);
        v37 = *v36;
        if ( v33 == *v36 )
          goto LABEL_30;
        v73 = v74;
      }
LABEL_10:
      v16 = v98;
      goto LABEL_11;
    }
LABEL_30:
    if ( v36 == (int *)(v34 + 8 * v32) )
      goto LABEL_10;
    v94 = v36[1];
    if ( v94 == -1 )
      goto LABEL_10;
  }
  if ( *(_DWORD *)(a1 + 312) )
  {
    v11 = *(_DWORD *)(a1 + 320);
    v12 = *(_QWORD *)(a1 + 304);
    if ( v11 )
    {
      v13 = v11 - 1;
      v14 = (v11 - 1) & (37 * v94);
      v15 = *(_DWORD *)(v12 + 4LL * v14);
      if ( v94 == v15 )
        goto LABEL_10;
      v63 = 1;
      while ( v15 != -1 )
      {
        v14 = v13 & (v63 + v14);
        v15 = *(_DWORD *)(v12 + 4LL * v14);
        if ( v94 == v15 )
          goto LABEL_10;
        ++v63;
      }
    }
  }
  LODWORD(v17) = 0;
  v18 = *(_QWORD *)(v7 + 40LL * (v8 + 5) + 24) / 8LL;
  v19 = (unsigned int)v18;
  if ( v8 + 8 == (*(_DWORD *)(a2 + 40) & 0xFFFFFF) )
    v17 = *(_QWORD *)(v7 + 40LL * (v8 + 7) + 24);
  v20 = a1;
  v21 = 0;
  v22 = v17;
  v23 = v8;
  while ( 1 )
  {
    v24 = v7 + 40LL * v21;
    if ( *(_BYTE *)v24 )
    {
LABEL_22:
      sub_308E6E0(v20, v94);
      v16 = v98;
      goto LABEL_11;
    }
    v25 = (__int16 ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v20 + 448) + 56LL)
                                  + 16LL * (*(_DWORD *)(v24 + 8) & 0x7FFFFFFF))
                      & 0xFFFFFFFFFFFFFFF8LL);
    if ( v25 == &off_4A2FC20 || v25 == &off_4A2FCE0 || v25 == &off_4A2FD40 || v25 == &off_4A2FA40 )
    {
      v26 = 0;
    }
    else
    {
      if ( v25 != &off_4A2F980 && v25 != &off_4A2FB60 )
        goto LABEL_22;
      v26 = 1;
    }
    v27 = &v96;
    *(__int64 *)((char *)v96.m128i_i64 + 4) = __PAIR64__(v26, v19);
    v28 = v98;
    v97.m128i_i64[0] = a2;
    v97.m128i_i32[2] = v21;
    v96.m128i_i32[0] = v22 + v18 * v21;
    v29 = (unsigned int)v99;
    if ( (unsigned __int64)(unsigned int)v99 + 1 > HIDWORD(v99) )
    {
      v90 = v18;
      v91 = v20;
      v92 = v19;
      v95 = v23;
      if ( v98 > &v96 || &v96 >= &v98[2 * (unsigned int)v99] )
      {
        v27 = &v96;
        sub_C8D5F0((__int64)&v98, v100, (unsigned int)v99 + 1LL, 0x20u, v23, v19);
        v28 = v98;
        v29 = (unsigned int)v99;
        v18 = v90;
        v20 = v91;
        v19 = v92;
        v23 = v95;
      }
      else
      {
        v38 = (char *)((char *)&v96 - (char *)v98);
        sub_C8D5F0((__int64)&v98, v100, (unsigned int)v99 + 1LL, 0x20u, v23, v19);
        v28 = v98;
        v29 = (unsigned int)v99;
        v23 = v95;
        v19 = v92;
        v20 = v91;
        v18 = v90;
        v27 = (__m128i *)&v38[(_QWORD)v98];
      }
    }
    ++v21;
    v30 = &v28[2 * v29];
    *v30 = _mm_loadu_si128(v27);
    v31 = _mm_loadu_si128(v27 + 1);
    LODWORD(v99) = v99 + 1;
    v30[1] = v31;
    if ( (unsigned int)v23 <= v21 )
      break;
    v7 = *(_QWORD *)(a2 + 32);
  }
  v39 = v20;
  v40 = v20 + 200;
  v41 = *(_DWORD *)(v39 + 224);
  if ( !v41 )
  {
    ++*(_QWORD *)(v39 + 200);
    goto LABEL_102;
  }
  v42 = *(_QWORD *)(v39 + 208);
  v43 = (v41 - 1) & (37 * v94);
  v44 = (int *)(v42 + 16LL * v43);
  v45 = *v44;
  if ( *v44 != v94 )
  {
    v64 = 1;
    v65 = 0;
    while ( v45 != 0x7FFFFFFF )
    {
      if ( v45 == 0x80000000 && !v65 )
        v65 = v44;
      v43 = (v41 - 1) & (v64 + v43);
      v44 = (int *)(v42 + 16LL * v43);
      v45 = *v44;
      if ( *v44 == v94 )
        goto LABEL_39;
      ++v64;
    }
    if ( !v65 )
      v65 = v44;
    v66 = *(_DWORD *)(v39 + 216);
    ++*(_QWORD *)(v39 + 200);
    v67 = v66 + 1;
    if ( 4 * (v66 + 1) < 3 * v41 )
    {
      if ( v41 - *(_DWORD *)(v39 + 220) - v67 > v41 >> 3 )
      {
LABEL_70:
        *(_DWORD *)(v39 + 216) = v67;
        if ( *v65 != 0x7FFFFFFF )
          --*(_DWORD *)(v39 + 220);
        *((_QWORD *)v65 + 1) = 0;
        *v65 = v94;
        v46 = (unsigned int ***)(v65 + 2);
        goto LABEL_73;
      }
      sub_308A1F0(v40, v41);
      v82 = *(_DWORD *)(v39 + 224);
      if ( v82 )
      {
        v83 = v82 - 1;
        v84 = *(_QWORD *)(v39 + 208);
        v85 = 1;
        v86 = v83 & (37 * v94);
        v67 = *(_DWORD *)(v39 + 216) + 1;
        v87 = 0;
        v65 = (int *)(v84 + 16LL * v86);
        v88 = *v65;
        if ( *v65 != v94 )
        {
          while ( v88 != 0x7FFFFFFF )
          {
            if ( v88 == 0x80000000 && !v87 )
              v87 = v65;
            v86 = v83 & (v85 + v86);
            v65 = (int *)(v84 + 16LL * v86);
            v88 = *v65;
            if ( *v65 == v94 )
              goto LABEL_70;
            ++v85;
          }
          if ( v87 )
            v65 = v87;
        }
        goto LABEL_70;
      }
LABEL_134:
      ++*(_DWORD *)(v39 + 216);
      BUG();
    }
LABEL_102:
    sub_308A1F0(v40, 2 * v41);
    v75 = *(_DWORD *)(v39 + 224);
    if ( v75 )
    {
      v76 = v75 - 1;
      v77 = *(_QWORD *)(v39 + 208);
      v67 = *(_DWORD *)(v39 + 216) + 1;
      v78 = v76 & (37 * v94);
      v65 = (int *)(v77 + 16LL * v78);
      v79 = *v65;
      if ( *v65 != v94 )
      {
        v80 = 1;
        v81 = 0;
        while ( v79 != 0x7FFFFFFF )
        {
          if ( !v81 && v79 == 0x80000000 )
            v81 = v65;
          v78 = v76 & (v80 + v78);
          v65 = (int *)(v77 + 16LL * v78);
          v79 = *v65;
          if ( *v65 == v94 )
            goto LABEL_70;
          ++v80;
        }
        if ( v81 )
          v65 = v81;
      }
      goto LABEL_70;
    }
    goto LABEL_134;
  }
LABEL_39:
  v46 = (unsigned int ***)(v44 + 2);
  if ( !*((_QWORD *)v44 + 1) )
  {
LABEL_73:
    v68 = (unsigned int **)sub_22077B0(0x210u);
    if ( v68 )
    {
      *v68 = (unsigned int *)(v68 + 2);
      v68[1] = (unsigned int *)0x1000000000LL;
    }
    *v46 = v68;
  }
  v47 = v98;
  v16 = &v98[2 * (unsigned int)v99];
  if ( v98 == v16 )
    goto LABEL_11;
  v93 = v39;
  v48 = &v98[2 * (unsigned int)v99];
  while ( 1 )
  {
    v49 = *v46;
    v50 = v47->m128i_i32[0];
    v51 = v47->m128i_u32[1];
    v52 = v47->m128i_i32[2];
    v53 = v47[1].m128i_i64[0];
    v54 = v47[1].m128i_i32[2];
    v96.m128i_i32[0] = v47->m128i_i32[0];
    *(__int64 *)((char *)v96.m128i_i64 + 4) = __PAIR64__(v52, v51);
    v97.m128i_i64[0] = v53;
    v97.m128i_i32[2] = v54;
    v55 = *((unsigned int *)v49 + 2);
    if ( !*((_DWORD *)v49 + 2) )
    {
      if ( !*((_DWORD *)v49 + 3) )
      {
        sub_C8D5F0((__int64)v49, v49 + 2, 1u, 0x20u, v55, v51);
        v55 = *((unsigned int *)v49 + 2);
      }
      v69 = (__m128i *)&(*v49)[8 * v55];
      *v69 = _mm_loadu_si128(&v96);
      v69[1] = _mm_loadu_si128(&v97);
      ++*((_DWORD *)v49 + 2);
      goto LABEL_44;
    }
    v56 = (unsigned __int64)*v49;
    v57 = **v49;
    if ( (int)v57 > v50 )
    {
      if ( (int)v57 < (int)v51 + v50 )
        goto LABEL_78;
      sub_308A020((__int64)v49, (__m128i *)*v49, &v96, v57, v55, v51);
      goto LABEL_44;
    }
    v58 = (__m128i *)(v56 + 32);
    v59 = (__m128i *)(v56 + 32 * v55);
    if ( v59 != (__m128i *)(v56 + 32) )
      break;
    v72 = (_DWORD)v57 == v50;
    if ( (int)v57 >= v50 )
    {
      v60 = (__m128i *)*v49;
      if ( !v72 )
      {
LABEL_92:
        v70 = v55 + 1;
        v71 = &v96;
        if ( v55 + 1 > (unsigned __int64)*((unsigned int *)v49 + 3) )
        {
          v89 = (__int64)(v49 + 2);
          if ( v56 > (unsigned __int64)&v96 || v58 <= &v96 )
          {
            sub_C8D5F0((__int64)v49, v49 + 2, v70, 0x20u, v89, v51);
            v71 = &v96;
            v58 = (__m128i *)&(*v49)[8 * *((unsigned int *)v49 + 2)];
          }
          else
          {
            sub_C8D5F0((__int64)v49, v49 + 2, v70, 0x20u, v89, v51);
            v71 = (__m128i *)((char *)&v96 + (_QWORD)*v49 - v56);
            v58 = (__m128i *)&(*v49)[8 * *((unsigned int *)v49 + 2)];
          }
        }
        *v58 = _mm_loadu_si128(v71);
        v58[1] = _mm_loadu_si128(v71 + 1);
        ++*((_DWORD *)v49 + 2);
        goto LABEL_44;
      }
LABEL_84:
      if ( (_DWORD)v51 != v60->m128i_i32[1] || v52 != v60->m128i_i32[2] )
        goto LABEL_78;
      goto LABEL_56;
    }
    v60 = (__m128i *)(unsigned int)(*(_DWORD *)(v56 + 4) + v57);
    if ( v50 < (int)v60 )
      goto LABEL_78;
LABEL_56:
    if ( v59 == v58 )
      goto LABEL_92;
    v62 = v58->m128i_i32[0];
    if ( v50 >= v58->m128i_i32[0] )
    {
      if ( v50 <= v62 )
      {
        if ( v50 == v62 && ((_DWORD)v51 != v58->m128i_i32[1] || v52 != v58->m128i_i32[2]) )
          goto LABEL_78;
      }
      else if ( v50 < v58->m128i_i32[1] + v62 )
      {
        goto LABEL_78;
      }
    }
    else if ( v62 < (int)v51 + v50 )
    {
      goto LABEL_78;
    }
    sub_308A020((__int64)v49, v58, &v96, (__int64)v60, v55, v51);
LABEL_44:
    v47 += 2;
    if ( v48 == v47 )
      goto LABEL_10;
  }
  while ( 1 )
  {
    v60 = v58 - 2;
    if ( v58->m128i_i32[0] >= v50 )
      break;
    if ( v59 == &v58[2] )
    {
      v60 = v58;
      v58 = (__m128i *)(v56 + 32 * v55);
      break;
    }
    v58 += 2;
  }
  v61 = v60->m128i_i32[0];
  if ( v50 <= v60->m128i_i32[0] || v50 >= v61 + v60->m128i_i32[1] )
  {
    if ( v50 >= v61 )
    {
      if ( v50 == v61 )
        goto LABEL_84;
    }
    else
    {
      v60 = (__m128i *)(unsigned int)(v51 + v50);
      if ( v61 < (int)v60 )
        goto LABEL_78;
    }
    goto LABEL_56;
  }
LABEL_78:
  sub_308E6E0(v93, v94);
  v16 = v98;
LABEL_11:
  if ( v16 != (__m128i *)v100 )
    _libc_free((unsigned __int64)v16);
}
