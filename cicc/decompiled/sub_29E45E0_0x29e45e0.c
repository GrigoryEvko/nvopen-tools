// Function: sub_29E45E0
// Address: 0x29e45e0
//
void __fastcall sub_29E45E0(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 *v4; // r13
  __int64 v5; // r14
  unsigned __int8 *i; // rdi
  __int64 v7; // rdi
  unsigned __int8 *v8; // rax
  __int64 v9; // rdx
  unsigned __int8 *v10; // r12
  unsigned __int8 v11; // al
  __int64 v12; // r8
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  char *v17; // r12
  int v18; // ecx
  unsigned __int8 *v19; // rdi
  __int64 v20; // rax
  unsigned __int8 *v21; // rdi
  __int64 v22; // rdi
  unsigned __int8 *v23; // rax
  __int64 v24; // rdx
  unsigned __int8 *v25; // r12
  unsigned __int8 v26; // al
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 v29; // rdi
  unsigned __int8 *v30; // rax
  __int64 v31; // rdx
  unsigned __int8 *v32; // rbx
  unsigned __int8 v33; // al
  __int64 v34; // r8
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // rbx
  __int64 v42; // rax
  unsigned int *v43; // rax
  __m128i *v44; // rax
  __m128i si128; // xmm0
  unsigned __int64 v46; // rax
  char *v47; // rax
  __m128i *v48; // rdi
  void *v49; // rax
  void *v50; // rsi
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  char *v53; // rcx
  size_t v54; // r8
  char *v55; // rax
  unsigned __int8 *v56; // r12
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // r14
  __int64 v61; // rbx
  __int64 v62; // rax
  unsigned __int64 v63; // [rsp+8h] [rbp-1A8h]
  size_t v64; // [rsp+8h] [rbp-1A8h]
  __int64 v65; // [rsp+8h] [rbp-1A8h]
  __int64 v66; // [rsp+10h] [rbp-1A0h]
  char *v68; // [rsp+28h] [rbp-188h]
  __int64 v70; // [rsp+40h] [rbp-170h]
  unsigned __int8 *v71; // [rsp+68h] [rbp-148h] BYREF
  unsigned __int64 v72; // [rsp+70h] [rbp-140h] BYREF
  unsigned __int64 v73; // [rsp+78h] [rbp-138h]
  __m128i v74; // [rsp+80h] [rbp-130h] BYREF
  void *src; // [rsp+90h] [rbp-120h]
  _BYTE *v76; // [rsp+98h] [rbp-118h]
  char *v77; // [rsp+A0h] [rbp-110h]
  __m128i *v78; // [rsp+B0h] [rbp-100h] BYREF
  unsigned __int64 v79; // [rsp+B8h] [rbp-F8h]
  __m128i v80; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned __int64 v81; // [rsp+D0h] [rbp-E0h]
  char *v82; // [rsp+D8h] [rbp-D8h]
  char *v83; // [rsp+E0h] [rbp-D0h]
  unsigned __int64 v84[2]; // [rsp+F0h] [rbp-C0h] BYREF
  _BYTE v85[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+120h] [rbp-90h]
  __int64 v87; // [rsp+128h] [rbp-88h]
  __int16 v88; // [rsp+130h] [rbp-80h]
  __int64 v89; // [rsp+138h] [rbp-78h]
  void **v90; // [rsp+140h] [rbp-70h]
  void **v91; // [rsp+148h] [rbp-68h]
  __int64 v92; // [rsp+150h] [rbp-60h]
  int v93; // [rsp+158h] [rbp-58h]
  __int16 v94; // [rsp+15Ch] [rbp-54h]
  char v95; // [rsp+15Eh] [rbp-52h]
  __int64 v96; // [rsp+160h] [rbp-50h]
  __int64 v97; // [rsp+168h] [rbp-48h]
  void *v98; // [rsp+170h] [rbp-40h] BYREF
  void *v99; // [rsp+178h] [rbp-38h] BYREF

  v3 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  v70 = v3;
  if ( *(_QWORD *)a3 == v3 )
    return;
  v4 = *(__int64 **)a3;
  do
  {
    v5 = *v4;
    for ( i = *(unsigned __int8 **)(*v4 - 32LL * (*(_DWORD *)(*v4 + 4) & 0x7FFFFFF));
          ;
          i = *(unsigned __int8 **)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)] )
    {
      v8 = sub_BD3990(i, v3);
      v7 = 23;
      v10 = v8;
      v11 = *v8;
      if ( v11 > 0x1Cu )
      {
        if ( v11 != 85 )
        {
          v7 = 2 * (unsigned int)(v11 != 34) + 21;
          goto LABEL_5;
        }
        v12 = *((_QWORD *)v10 - 4);
        v7 = 21;
        if ( v12 )
        {
          if ( !*(_BYTE *)v12 && *(_QWORD *)(v12 + 24) == *((_QWORD *)v10 + 10) )
            break;
        }
      }
LABEL_5:
      if ( !(unsigned __int8)sub_3108CA0(v7) )
        goto LABEL_13;
LABEL_6:
      ;
    }
    v13 = sub_3108960(*((_QWORD *)v10 - 4), v3, v9);
    if ( (unsigned __int8)sub_3108CA0(v13) )
      goto LABEL_6;
LABEL_13:
    v71 = v10;
    v14 = sub_BD5C60(v5);
    v91 = &v99;
    v89 = v14;
    v92 = 0;
    v90 = &v98;
    v93 = 0;
    v98 = &unk_49DA100;
    v94 = 512;
    v95 = 7;
    v96 = 0;
    v97 = 0;
    v86 = 0;
    v87 = 0;
    v88 = 0;
    v99 = &unk_49DA0B0;
    v15 = *(_QWORD *)(v5 + 40);
    v16 = *(_QWORD *)(v5 + 24);
    v84[0] = (unsigned __int64)v85;
    v3 = 0x200000000LL;
    v84[1] = 0x200000000LL;
    v17 = (char *)(v16 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v17 == (char *)(v15 + 48) )
      goto LABEL_33;
    while ( 1 )
    {
      v18 = (unsigned __int8)*(v17 - 24);
      if ( (unsigned int)(v18 - 67) > 0xC )
        break;
      v17 = (char *)(*(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (char *)(v15 + 48) == v17 )
      {
        if ( a2 != 1 )
          goto LABEL_34;
        goto LABEL_78;
      }
    }
    if ( (_BYTE)v18 != 85 )
      goto LABEL_33;
    v68 = v17 - 24;
    v19 = (unsigned __int8 *)(v17 - 24);
    v20 = *((_QWORD *)v17 - 7);
    if ( !v20 || *(_BYTE *)v20 )
    {
      v19 = (unsigned __int8 *)(v17 - 24);
      goto LABEL_41;
    }
    if ( *(_QWORD *)(v20 + 24) != *((_QWORD *)v17 + 7) || (*(_BYTE *)(v20 + 33) & 0x20) == 0 )
    {
      while ( 1 )
      {
LABEL_41:
        v30 = sub_BD3990(v19, 0x200000000LL);
        v29 = 23;
        v32 = v30;
        v33 = *v30;
        if ( v33 <= 0x1Cu )
          goto LABEL_39;
        if ( v33 != 85 )
          break;
        v34 = *((_QWORD *)v32 - 4);
        v29 = 21;
        if ( !v34 || *(_BYTE *)v34 || *(_QWORD *)(v34 + 24) != *((_QWORD *)v32 + 10) )
          goto LABEL_39;
        v35 = sub_3108960(*((_QWORD *)v32 - 4), 0x200000000LL, v31);
        if ( !(unsigned __int8)sub_3108CA0(v35) )
        {
LABEL_47:
          if ( v71 == v32 )
          {
            if ( *(_BYTE *)(**(_QWORD **)(*((_QWORD *)v17 + 7) + 16LL) + 8LL) == 7
              || *(v17 - 17) >= 0
              || ((v57 = sub_BD2BC0((__int64)v68), *(v17 - 17) >= 0)
                ? (v59 = (v57 + v58) >> 4)
                : (LODWORD(v59) = (v57 + v58 - sub_BD2BC0((__int64)v68)) >> 4),
                  !(_DWORD)v59) )
            {
LABEL_49:
              if ( *(char *)(a1 + 7) < 0 )
              {
                v36 = sub_BD2BC0(a1);
                v38 = v36 + v37;
                if ( *(char *)(a1 + 7) >= 0 )
                  v39 = v38 >> 4;
                else
                  LODWORD(v39) = (v38 - sub_BD2BC0(a1)) >> 4;
                if ( (_DWORD)v39 )
                {
                  v40 = 0;
                  v41 = 16LL * (unsigned int)v39;
                  while ( 1 )
                  {
                    v42 = 0;
                    if ( *(char *)(a1 + 7) < 0 )
                      v42 = sub_BD2BC0(a1);
                    v43 = (unsigned int *)(v40 + v42);
                    if ( *(_DWORD *)(*(_QWORD *)v43 + 8LL) == 6 )
                      break;
                    v40 += 16;
                    if ( v40 == v41 )
                      goto LABEL_59;
                  }
                  v66 = *(_QWORD *)(a1 + 32 * (v43[2] - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
                }
              }
LABEL_59:
              v72 = 22;
              v78 = &v80;
              v44 = (__m128i *)sub_22409D0((__int64)&v78, &v72, 0);
              si128 = _mm_load_si128((const __m128i *)&xmmword_439B070);
              v78 = v44;
              v80.m128i_i64[0] = v72;
              v44[1].m128i_i32[0] = 1633903717;
              v44[1].m128i_i16[2] = 27756;
              *v44 = si128;
              v79 = v72;
              v78->m128i_i8[v72] = 0;
              v72 = (unsigned __int64)&v74;
              if ( v78 == &v80 )
              {
                v74 = _mm_load_si128(&v80);
              }
              else
              {
                v72 = (unsigned __int64)v78;
                v74.m128i_i64[0] = v80.m128i_i64[0];
              }
              v46 = v79;
              v78 = &v80;
              v79 = 0;
              v73 = v46;
              v80.m128i_i8[0] = 0;
              src = 0;
              v76 = 0;
              v77 = 0;
              v47 = (char *)sub_22077B0(8u);
              v48 = v78;
              src = v47;
              v77 = v47 + 8;
              *(_QWORD *)v47 = v66;
              v76 = v47 + 8;
              if ( v48 != &v80 )
                j_j___libc_free_0((unsigned __int64)v48);
              v78 = &v80;
              sub_29E0270((__int64 *)&v78, (_BYTE *)v72, v72 + v73);
              v49 = v76;
              v50 = src;
              v81 = 0;
              v82 = 0;
              v83 = 0;
              v51 = v76 - (_BYTE *)src;
              if ( v76 == src )
              {
                v54 = 0;
                v53 = 0;
              }
              else
              {
                if ( v51 > 0x7FFFFFFFFFFFFFF8LL )
                  sub_4261EA(&v78, src, v51);
                v63 = v76 - (_BYTE *)src;
                v52 = sub_22077B0(v76 - (_BYTE *)src);
                v50 = src;
                v51 = v63;
                v53 = (char *)v52;
                v49 = v76;
                v54 = v76 - (_BYTE *)src;
              }
              v81 = (unsigned __int64)v53;
              v82 = v53;
              v83 = &v53[v51];
              if ( v50 != v49 )
              {
                v64 = v54;
                v55 = (char *)memmove(v53, v50, v54);
                v54 = v64;
                v53 = v55;
              }
              v82 = &v53[v54];
              v56 = sub_B56B10((__int64)v68, 6, (char *)&v78, (__int64)v17, 0);
              if ( v81 )
                j_j___libc_free_0(v81);
              if ( v78 != &v80 )
                j_j___libc_free_0((unsigned __int64)v78);
              sub_B47C00((__int64)v56, (__int64)v68, 0, 0);
              v3 = (__int64)v56;
              sub_BD84D0((__int64)v68, (__int64)v56);
              sub_B43D60(v68);
              if ( src )
              {
                v3 = v77 - (_BYTE *)src;
                j_j___libc_free_0((unsigned __int64)src);
              }
              if ( (__m128i *)v72 != &v74 )
              {
                v3 = v74.m128i_i64[0] + 1;
                j_j___libc_free_0(v72);
              }
              goto LABEL_34;
            }
            v65 = v5;
            v60 = 0;
            v61 = 16LL * (unsigned int)v59;
            while ( 1 )
            {
              v62 = 0;
              if ( *(v17 - 17) < 0 )
                v62 = sub_BD2BC0((__int64)v68);
              if ( *(_DWORD *)(*(_QWORD *)(v62 + v60) + 8LL) == 6 )
                break;
              v60 += 16;
              if ( v61 == v60 )
                goto LABEL_49;
            }
            v5 = v65;
          }
LABEL_33:
          if ( a2 != 1 )
            goto LABEL_34;
LABEL_78:
          sub_D5F1F0((__int64)v84, v5);
          LOWORD(v81) = 257;
          v3 = 268;
          HIDWORD(v72) = 0;
          sub_B33D10((__int64)v84, 0x10Cu, 0, 0, (int)&v71, 1, (unsigned int)v72, (__int64)&v78);
          goto LABEL_34;
        }
LABEL_40:
        v19 = *(unsigned __int8 **)&v32[-32 * (*((_DWORD *)v32 + 1) & 0x7FFFFFF)];
      }
      v29 = 2 * (unsigned int)(v33 != 34) + 21;
LABEL_39:
      if ( !(unsigned __int8)sub_3108CA0(v29) )
        goto LABEL_47;
      goto LABEL_40;
    }
    if ( *(_DWORD *)(v20 + 36) != 258 )
      goto LABEL_33;
    v3 = 0;
    if ( !(unsigned __int8)sub_BD3610((__int64)v19, 0) )
      goto LABEL_33;
    v21 = *(unsigned __int8 **)&v19[-32 * (*((_DWORD *)v17 - 5) & 0x7FFFFFF)];
    while ( 2 )
    {
      v23 = sub_BD3990(v21, 0);
      v22 = 23;
      v25 = v23;
      v26 = *v23;
      if ( v26 <= 0x1Cu )
        goto LABEL_24;
      if ( v26 != 85 )
      {
        v22 = 2 * (unsigned int)(v26 != 34) + 21;
LABEL_24:
        if ( !(unsigned __int8)sub_3108CA0(v22) )
          goto LABEL_32;
LABEL_25:
        v21 = *(unsigned __int8 **)&v25[-32 * (*((_DWORD *)v25 + 1) & 0x7FFFFFF)];
        continue;
      }
      break;
    }
    v27 = *((_QWORD *)v25 - 4);
    v22 = 21;
    if ( !v27 || *(_BYTE *)v27 || *(_QWORD *)(v27 + 24) != *((_QWORD *)v25 + 10) )
      goto LABEL_24;
    v28 = sub_3108960(*((_QWORD *)v25 - 4), 0, v24);
    if ( (unsigned __int8)sub_3108CA0(v28) )
      goto LABEL_25;
LABEL_32:
    if ( v71 != v25 )
      goto LABEL_33;
    if ( a2 != 1 )
    {
      sub_D5F1F0((__int64)v84, (__int64)v68);
      HIDWORD(v72) = 0;
      v3 = 267;
      LOWORD(v81) = 257;
      sub_B33D10((__int64)v84, 0x10Bu, 0, 0, (int)&v71, 1, (unsigned int)v72, (__int64)&v78);
    }
    sub_B43D60(v68);
LABEL_34:
    nullsub_61();
    v98 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v84[0] != v85 )
      _libc_free(v84[0]);
    ++v4;
  }
  while ( (__int64 *)v70 != v4 );
}
