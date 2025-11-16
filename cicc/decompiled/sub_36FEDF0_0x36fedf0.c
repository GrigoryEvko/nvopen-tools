// Function: sub_36FEDF0
// Address: 0x36fedf0
//
void __fastcall sub_36FEDF0(
        unsigned __int64 a1,
        __m128i **a2,
        __int64 *a3,
        __int64 a4,
        _BYTE *a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8)
{
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r13
  char *v13; // r15
  unsigned __int64 v14; // rbx
  char *v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r14
  unsigned __int64 v20; // r12
  char *v21; // rcx
  __int64 v22; // r12
  __m128i **v23; // r13
  size_t v24; // r12
  __int64 v25; // rax
  __m128i **v26; // rbx
  size_t v27; // r14
  unsigned __int64 v28; // rbx
  char *v29; // rcx
  const void *v30; // rsi
  __int64 v31; // rbx
  __int64 v32; // rax
  int v33; // eax
  signed __int64 v34; // r15
  __int64 v35; // rdx
  char *v36; // rbx
  char *v37; // rax
  int v38; // edx
  __m128i *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // r15
  __m128i *v42; // rax
  __int64 v43; // rax
  __m128i **v44; // r15
  size_t v45; // rbx
  unsigned __int64 *v46; // rax
  __int64 v47; // rax
  __m128i **v48; // rax
  unsigned __int64 v49; // rbx
  __int64 v50; // rax
  char *v51; // rcx
  size_t v52; // r15
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rbx
  _QWORD *v56; // r15
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // r8
  __int64 v59; // rax
  __int64 v60; // r12
  __int64 v61; // rbx
  _QWORD *v62; // rdi
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // rdi
  unsigned __int64 *v65; // rbx
  unsigned __int64 *v66; // r12
  unsigned __int64 v67; // rdi
  __int64 v68; // rsi
  unsigned __int64 v69; // r12
  char *v70; // r14
  char *v71; // rax
  unsigned __int64 *v72; // r15
  unsigned __int64 *v73; // r15
  unsigned __int64 v74; // rax
  __int64 *v75; // rdi
  __int64 v78; // [rsp+18h] [rbp-F8h]
  __int64 v79; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v80; // [rsp+28h] [rbp-E8h]
  __int64 n; // [rsp+38h] [rbp-D8h]
  __m128i *v82; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v83; // [rsp+48h] [rbp-C8h]
  __m128i v84; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int64 v85[2]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v86[2]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 *v87; // [rsp+80h] [rbp-90h] BYREF
  __int64 v88; // [rsp+88h] [rbp-88h]
  _QWORD v89[2]; // [rsp+90h] [rbp-80h] BYREF
  __int64 v90; // [rsp+A0h] [rbp-70h]
  _BYTE *v91; // [rsp+A8h] [rbp-68h]
  char *v92; // [rsp+B0h] [rbp-60h]
  char *v93; // [rsp+B8h] [rbp-58h]
  __int64 v94; // [rsp+C0h] [rbp-50h]
  __int64 v95; // [rsp+C8h] [rbp-48h]
  char v96; // [rsp+D0h] [rbp-40h]

  v78 = (__int64)a2;
  v10 = a3[1] - *a3;
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 16) = 4;
  v11 = 0xCCCCCCCCCCCCCCCDLL * (v10 >> 4);
  *(_QWORD *)a1 = &unk_4A399F0;
  if ( v10 < 0 )
    goto LABEL_134;
  *(_QWORD *)(a1 + 24) = 0;
  v12 = a1;
  v13 = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  if ( v11 )
  {
    v14 = 0x6666666666666668LL * (v10 >> 4);
    v15 = (char *)sub_22077B0(v14);
    v13 = &v15[v14];
    *(_QWORD *)(a1 + 24) = v15;
    a1 = (unsigned __int64)v15;
    *(_QWORD *)(v12 + 40) = &v15[v14];
    if ( v15 != &v15[v14] )
    {
      a2 = 0;
      memset(v15, 0, v14);
    }
  }
  *(_QWORD *)(v12 + 32) = v13;
  *(_QWORD *)(v12 + 48) = 0;
  *(_QWORD *)(v12 + 56) = 0;
  *(_QWORD *)(v12 + 64) = 0;
  *(_QWORD *)v12 = &unk_4A3C580;
  *(_DWORD *)(v12 + 72) = -1;
  v16 = a3[1] - *a3;
  *(_QWORD *)(v12 + 80) = 0;
  *(_QWORD *)(v12 + 88) = 0;
  *(_QWORD *)(v12 + 96) = 0;
  if ( v16 )
  {
    if ( v16 > 0x7FFFFFFFFFFFFFD0LL )
      goto LABEL_133;
    v17 = sub_22077B0(v16);
  }
  else
  {
    v16 = 0;
    v17 = 0;
  }
  *(_QWORD *)(v12 + 80) = v17;
  *(_QWORD *)(v12 + 88) = v17;
  *(_QWORD *)(v12 + 96) = v17 + v16;
  v18 = a3[1];
  if ( v18 != *a3 )
  {
    v80 = v12;
    v19 = *a3;
    v79 = a4;
    while ( !v17 )
    {
LABEL_15:
      v19 += 80;
      v17 += 80;
      if ( v18 == v19 )
      {
        v12 = v80;
        a4 = v79;
        goto LABEL_29;
      }
    }
    a1 = v17 + 16;
    *(_QWORD *)v17 = v17 + 16;
    v23 = *(__m128i ***)v19;
    v24 = *(_QWORD *)(v19 + 8);
    if ( v24 + *(_QWORD *)v19 && !v23 )
      goto LABEL_135;
    v87 = *(unsigned __int64 **)(v19 + 8);
    if ( v24 > 0xF )
    {
      v25 = sub_22409D0(v17, (unsigned __int64 *)&v87, 0);
      *(_QWORD *)v17 = v25;
      a1 = v25;
      *(_QWORD *)(v17 + 16) = v87;
    }
    else
    {
      if ( v24 == 1 )
      {
        *(_BYTE *)(v17 + 16) = *(_BYTE *)v23;
LABEL_22:
        *(_QWORD *)(v17 + 8) = v24;
        *(_BYTE *)(a1 + v24) = 0;
        *(_DWORD *)(v17 + 32) = *(_DWORD *)(v19 + 32);
        *(_DWORD *)(v17 + 36) = *(_DWORD *)(v19 + 36);
        v11 = *(_QWORD *)(v19 + 48) - *(_QWORD *)(v19 + 40);
        *(_QWORD *)(v17 + 40) = 0;
        *(_QWORD *)(v17 + 48) = 0;
        *(_QWORD *)(v17 + 56) = 0;
        if ( v11 )
        {
          v20 = v11;
          if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_133;
          v21 = (char *)sub_22077B0(v11);
        }
        else
        {
          v20 = 0;
          v21 = 0;
        }
        v11 = (unsigned __int64)&v21[v20];
        *(_QWORD *)(v17 + 40) = v21;
        *(_QWORD *)(v17 + 48) = v21;
        *(_QWORD *)(v17 + 56) = &v21[v20];
        a2 = *(__m128i ***)(v19 + 40);
        v22 = *(_QWORD *)(v19 + 48) - (_QWORD)a2;
        if ( *(__m128i ***)(v19 + 48) != a2 )
          v21 = (char *)memmove(v21, a2, *(_QWORD *)(v19 + 48) - (_QWORD)a2);
        *(_QWORD *)(v17 + 48) = &v21[v22];
        *(_QWORD *)(v17 + 64) = *(_QWORD *)(v19 + 64);
        *(_QWORD *)(v17 + 72) = *(_QWORD *)(v19 + 72);
        goto LABEL_15;
      }
      if ( !v24 )
        goto LABEL_22;
    }
    a2 = v23;
    memcpy((void *)a1, v23, v24);
    v24 = (size_t)v87;
    a1 = *(_QWORD *)v17;
    goto LABEL_22;
  }
LABEL_29:
  a1 = v12 + 120;
  *(_QWORD *)(v12 + 88) = v17;
  *(_QWORD *)(v12 + 104) = v12 + 120;
  v26 = *(__m128i ***)a4;
  v27 = *(_QWORD *)(a4 + 8);
  if ( v27 + *(_QWORD *)a4 && !v26 )
LABEL_135:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v87 = *(unsigned __int64 **)(a4 + 8);
  if ( v27 > 0xF )
  {
    v43 = sub_22409D0(v12 + 104, (unsigned __int64 *)&v87, 0);
    *(_QWORD *)(v12 + 104) = v43;
    a1 = v43;
    *(_QWORD *)(v12 + 120) = v87;
LABEL_66:
    a2 = v26;
    memcpy((void *)a1, v26, v27);
    v27 = (size_t)v87;
    a1 = *(_QWORD *)(v12 + 104);
    goto LABEL_34;
  }
  if ( v27 == 1 )
  {
    *(_BYTE *)(v12 + 120) = *(_BYTE *)v26;
    goto LABEL_34;
  }
  if ( v27 )
    goto LABEL_66;
LABEL_34:
  *(_QWORD *)(v12 + 112) = v27;
  *(_BYTE *)(a1 + v27) = 0;
  *(_QWORD *)(v12 + 136) = *(_QWORD *)(a4 + 32);
  v28 = *(_QWORD *)(a4 + 48) - *(_QWORD *)(a4 + 40);
  *(_QWORD *)(v12 + 144) = 0;
  *(_QWORD *)(v12 + 152) = 0;
  *(_QWORD *)(v12 + 160) = 0;
  if ( v28 )
  {
    if ( v28 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_133;
    v29 = (char *)sub_22077B0(v28);
  }
  else
  {
    v29 = 0;
  }
  *(_QWORD *)(v12 + 144) = v29;
  *(_QWORD *)(v12 + 160) = &v29[v28];
  *(_QWORD *)(v12 + 152) = v29;
  v30 = *(const void **)(a4 + 40);
  v31 = *(_QWORD *)(a4 + 48) - (_QWORD)v30;
  if ( *(const void **)(a4 + 48) != v30 )
    v29 = (char *)memmove(v29, v30, *(_QWORD *)(a4 + 48) - (_QWORD)v30);
  *(_QWORD *)(v12 + 152) = &v29[v31];
  *(_QWORD *)(v12 + 168) = *(_QWORD *)(a4 + 64);
  v32 = *(_QWORD *)(a4 + 72);
  *(_DWORD *)(v12 + 184) = 0;
  *(_QWORD *)(v12 + 176) = v32;
  *(_QWORD *)(v12 + 192) = sub_2241E40();
  LOWORD(v90) = 261;
  v87 = (unsigned __int64 *)a7;
  v88 = a8;
  v33 = sub_C834C0((__int64)&v87, (int *)(v12 + 72), 0, 0);
  v34 = *(_QWORD *)(v12 + 168) * *(_QWORD *)(v12 + 176);
  *(_DWORD *)(v12 + 200) = v33;
  *(_QWORD *)(v12 + 208) = v35;
  if ( v34 < 0 )
LABEL_134:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  *(_QWORD *)(v12 + 216) = 0;
  v36 = 0;
  *(_QWORD *)(v12 + 224) = 0;
  *(_QWORD *)(v12 + 232) = 0;
  if ( v34 )
  {
    v37 = (char *)sub_22077B0(v34);
    v36 = &v37[v34];
    *(_QWORD *)(v12 + 216) = v37;
    *(_QWORD *)(v12 + 232) = &v37[v34];
    memset(v37, 0, v34);
  }
  v38 = *(_DWORD *)(v12 + 200);
  *(_QWORD *)(v12 + 224) = v36;
  *(_QWORD *)(v12 + 240) = 0;
  if ( v38 )
  {
    (*(void (__fastcall **)(unsigned __int64 *))(**(_QWORD **)(v12 + 208) + 32LL))(v85);
    v39 = (__m128i *)sub_2241130(v85, 0, 0, "Cannot open inbound file: ", 0x1Au);
    v82 = &v84;
    if ( (__m128i *)v39->m128i_i64[0] == &v39[1] )
    {
      v84 = _mm_loadu_si128(v39 + 1);
    }
    else
    {
      v82 = (__m128i *)v39->m128i_i64[0];
      v84.m128i_i64[0] = v39[1].m128i_i64[0];
    }
    v83 = v39->m128i_i64[1];
    v39->m128i_i64[0] = (__int64)v39[1].m128i_i64;
    v39->m128i_i64[1] = 0;
    v39[1].m128i_i8[0] = 0;
    LOWORD(v90) = 260;
    v87 = (unsigned __int64 *)&v82;
    sub_B6ECE0(v78, (__int64)&v87);
    if ( v82 != &v84 )
      j_j___libc_free_0((unsigned __int64)v82);
    if ( (_QWORD *)v85[0] != v86 )
      j_j___libc_free_0(v85[0]);
    return;
  }
  a1 = 96;
  v40 = sub_22077B0(0x60u);
  v41 = v40;
  if ( v40 )
  {
    a1 = v40;
    sub_CB7040(v40, a5, a6, v12 + 184);
  }
  v11 = *(unsigned int *)(v12 + 184);
  if ( (_DWORD)v11 )
  {
    (*(void (__fastcall **)(unsigned __int64 *))(**(_QWORD **)(v12 + 192) + 32LL))(v85);
    v42 = (__m128i *)sub_2241130(v85, 0, 0, "Cannot open outbound file: ", 0x1Bu);
    v82 = &v84;
    if ( (__m128i *)v42->m128i_i64[0] == &v42[1] )
    {
      v84 = _mm_loadu_si128(v42 + 1);
    }
    else
    {
      v82 = (__m128i *)v42->m128i_i64[0];
      v84.m128i_i64[0] = v42[1].m128i_i64[0];
    }
    v83 = v42->m128i_i64[1];
    v42->m128i_i64[0] = (__int64)v42[1].m128i_i64;
    v42->m128i_i64[1] = 0;
    v42[1].m128i_i8[0] = 0;
    LOWORD(v90) = 260;
    v87 = (unsigned __int64 *)&v82;
    sub_B6ECE0(v78, (__int64)&v87);
    if ( v82 != &v84 )
      j_j___libc_free_0((unsigned __int64)v82);
    if ( (_QWORD *)v85[0] != v86 )
      j_j___libc_free_0(v85[0]);
    if ( v41 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v41 + 8LL))(v41);
    return;
  }
  v82 = (__m128i *)v41;
  v44 = *(__m128i ***)a4;
  v45 = *(_QWORD *)(a4 + 8);
  v87 = v89;
  if ( (__m128i **)((char *)v44 + v45) && !v44 )
    goto LABEL_135;
  v85[0] = v45;
  if ( v45 > 0xF )
  {
    v87 = (unsigned __int64 *)sub_22409D0((__int64)&v87, v85, 0);
    a1 = (unsigned __int64)v87;
    v89[0] = v85[0];
LABEL_121:
    memcpy((void *)a1, v44, v45);
    v45 = v85[0];
    v46 = v87;
    goto LABEL_72;
  }
  if ( v45 == 1 )
  {
    LOBYTE(v89[0]) = *(_BYTE *)v44;
    v46 = v89;
    goto LABEL_72;
  }
  if ( v45 )
  {
    a1 = (unsigned __int64)v89;
    goto LABEL_121;
  }
  v46 = v89;
LABEL_72:
  v88 = v45;
  *((_BYTE *)v46 + v45) = 0;
  v47 = *(_QWORD *)(a4 + 32);
  a2 = *(__m128i ***)(a4 + 40);
  v91 = 0;
  v90 = v47;
  v48 = *(__m128i ***)(a4 + 48);
  v92 = 0;
  v93 = 0;
  v49 = (char *)v48 - (char *)a2;
  if ( v48 == a2 )
  {
    v52 = 0;
    v51 = 0;
    goto LABEL_75;
  }
  if ( v49 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_133:
    sub_4261EA(a1, a2, v11);
  v50 = sub_22077B0(v49);
  a2 = *(__m128i ***)(a4 + 40);
  v51 = (char *)v50;
  v48 = *(__m128i ***)(a4 + 48);
  v52 = (char *)v48 - (char *)a2;
LABEL_75:
  v91 = v51;
  v92 = v51;
  v93 = &v51[v49];
  if ( a2 != v48 )
    v51 = (char *)memmove(v51, a2, v52);
  v53 = *(_QWORD *)(a4 + 64);
  v96 = 1;
  v92 = &v51[v52];
  v94 = v53;
  v95 = *(_QWORD *)(a4 + 72);
  v54 = sub_22077B0(0xB0u);
  v55 = v54;
  if ( v54 )
  {
    a2 = &v82;
    sub_3700680(v54, &v82, v12 + 80, a4, 0, &v87);
  }
  if ( v96 )
  {
    v96 = 0;
    if ( v91 )
    {
      a2 = (__m128i **)(v93 - v91);
      j_j___libc_free_0((unsigned __int64)v91);
    }
    if ( v87 != v89 )
    {
      a2 = (__m128i **)(v89[0] + 1LL);
      j_j___libc_free_0((unsigned __int64)v87);
    }
  }
  if ( v82 )
    (*(void (__fastcall **)(__m128i *))(v82->m128i_i64[0] + 8))(v82);
  v56 = *(_QWORD **)(v12 + 240);
  *(_QWORD *)(v12 + 240) = v55;
  if ( v56 )
  {
    v57 = v56[18];
    if ( (_QWORD *)v57 != v56 + 20 )
    {
      a2 = (__m128i **)(v56[20] + 1LL);
      j_j___libc_free_0(v57);
    }
    v58 = v56[15];
    if ( *((_DWORD *)v56 + 33) )
    {
      v59 = *((unsigned int *)v56 + 32);
      if ( (_DWORD)v59 )
      {
        v60 = 8 * v59;
        v61 = 0;
        do
        {
          v62 = *(_QWORD **)(v58 + v61);
          if ( v62 && v62 != (_QWORD *)-8LL )
          {
            a2 = (__m128i **)(*v62 + 17LL);
            sub_C7D6A0((__int64)v62, (__int64)a2, 8);
            v58 = v56[15];
          }
          v61 += 8;
        }
        while ( v61 != v60 );
      }
    }
    _libc_free(v58);
    v63 = v56[9];
    if ( v63 )
    {
      a2 = (__m128i **)(v56[11] - v63);
      j_j___libc_free_0(v63);
    }
    v64 = v56[4];
    if ( (_QWORD *)v64 != v56 + 6 )
    {
      a2 = (__m128i **)(v56[6] + 1LL);
      j_j___libc_free_0(v64);
    }
    v65 = (unsigned __int64 *)v56[2];
    v66 = (unsigned __int64 *)v56[1];
    if ( v65 != v66 )
    {
      do
      {
        v67 = v66[5];
        if ( v67 )
        {
          a2 = (__m128i **)(v66[7] - v67);
          j_j___libc_free_0(v67);
        }
        if ( (unsigned __int64 *)*v66 != v66 + 2 )
        {
          a2 = (__m128i **)(v66[2] + 1);
          j_j___libc_free_0(*v66);
        }
        v66 += 10;
      }
      while ( v65 != v66 );
      v66 = (unsigned __int64 *)v56[1];
    }
    if ( v66 )
    {
      a2 = (__m128i **)(v56[3] - (_QWORD)v66);
      j_j___libc_free_0((unsigned __int64)v66);
    }
    if ( *v56 )
      (*(void (__fastcall **)(_QWORD, __m128i **))(*(_QWORD *)*v56 + 8LL))(*v56, a2);
    j_j___libc_free_0((unsigned __int64)v56);
  }
  v68 = *(_QWORD *)(v12 + 80);
  v69 = 0;
  if ( *(_QWORD *)(v12 + 88) != v68 )
  {
    do
    {
      v72 = *(unsigned __int64 **)(v12 + 56);
      v74 = *(_QWORD *)(v68 + 80 * v69 + 72) * *(_QWORD *)(v68 + 80 * v69 + 64);
      v87 = (unsigned __int64 *)v74;
      if ( v72 == *(unsigned __int64 **)(v12 + 64) )
      {
        sub_36FEBA0((unsigned __int64 **)(v12 + 48), v72, &v87);
        v73 = *(unsigned __int64 **)(v12 + 56);
      }
      else
      {
        if ( v72 )
        {
          if ( v74 > 0xFFFFFFFFFFFFFFFLL )
            goto LABEL_134;
          *v72 = 0;
          v70 = 0;
          v72[1] = 0;
          v72[2] = 0;
          if ( v74 )
          {
            n = 8 * v74;
            v71 = (char *)sub_22077B0(8 * v74);
            *v72 = (unsigned __int64)v71;
            v70 = &v71[n];
            v72[2] = (unsigned __int64)&v71[n];
            if ( v71 != &v71[n] )
              memset(v71, 0, n);
          }
          v72[1] = (unsigned __int64)v70;
          v72 = *(unsigned __int64 **)(v12 + 56);
        }
        v73 = v72 + 3;
        *(_QWORD *)(v12 + 56) = v73;
      }
      *(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v69) = *(v73 - 3);
      v68 = *(_QWORD *)(v12 + 80);
      ++v69;
    }
    while ( v69 < 0xCCCCCCCCCCCCCCCDLL * ((*(_QWORD *)(v12 + 88) - v68) >> 4) );
  }
  v75 = **(__int64 ***)(v12 + 240);
  if ( v75[4] != v75[2] )
    sub_CB5AE0(v75);
}
