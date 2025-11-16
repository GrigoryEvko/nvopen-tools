// Function: sub_2E59B70
// Address: 0x2e59b70
//
__int64 __fastcall sub_2E59B70(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  char *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r12d
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // eax
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // rcx
  _QWORD *v24; // rax
  _DWORD *v25; // rdx
  __int64 v26; // r9
  _BOOL4 v27; // esi
  unsigned int v28; // ecx
  __int64 v29; // rbx
  unsigned __int64 v30; // rdx
  int v31; // r10d
  unsigned __int64 v32; // rdi
  int v33; // r8d
  int v34; // r8d
  __int64 v35; // r9
  unsigned int v36; // ecx
  int v37; // r8d
  char *v38; // rdx
  __int64 i; // r11
  unsigned __int64 v40; // r11
  int v41; // eax
  unsigned int v42; // r10d
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // r11
  unsigned __int64 *v46; // rax
  __int64 v47; // r9
  unsigned __int8 *v48; // rsi
  unsigned __int64 *v49; // rdx
  __int64 v50; // rdi
  _BYTE *v51; // rax
  __int64 v52; // rbx
  void *v53; // rax
  __m128i *v54; // rdx
  __int64 v55; // r8
  __int64 v56; // r12
  _QWORD *v57; // rax
  __m128i *v58; // rdx
  __int64 v59; // r8
  __m128i si128; // xmm0
  unsigned int v61; // eax
  _QWORD *v62; // r12
  _QWORD *v63; // r13
  unsigned __int64 v64; // r14
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rdi
  size_t v67; // rdx
  int v68; // eax
  size_t v69; // rdx
  __int64 v70; // rcx
  _BOOL8 v71; // [rsp+10h] [rbp-150h]
  int v72; // [rsp+10h] [rbp-150h]
  int v73; // [rsp+18h] [rbp-148h]
  _BOOL8 v74; // [rsp+18h] [rbp-148h]
  __int64 v75; // [rsp+18h] [rbp-148h]
  _BOOL8 v76; // [rsp+18h] [rbp-148h]
  __int64 v77; // [rsp+20h] [rbp-140h]
  __int64 v78; // [rsp+20h] [rbp-140h]
  _BOOL8 v79; // [rsp+20h] [rbp-140h]
  __int64 v80; // [rsp+20h] [rbp-140h]
  __int64 v81; // [rsp+20h] [rbp-140h]
  __int64 v82; // [rsp+28h] [rbp-138h]
  __int64 v83; // [rsp+28h] [rbp-138h]
  __int64 v84; // [rsp+28h] [rbp-138h]
  unsigned int v85; // [rsp+30h] [rbp-130h]
  unsigned int v86; // [rsp+30h] [rbp-130h]
  unsigned int v87; // [rsp+30h] [rbp-130h]
  char v88; // [rsp+30h] [rbp-130h]
  unsigned __int8 v90; // [rsp+38h] [rbp-128h]
  __int64 v91; // [rsp+38h] [rbp-128h]
  __int64 v92; // [rsp+40h] [rbp-120h]
  __int64 v93; // [rsp+48h] [rbp-118h]
  __int64 v94; // [rsp+58h] [rbp-108h] BYREF
  __int64 v95; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v96; // [rsp+68h] [rbp-F8h]
  __int64 v97; // [rsp+70h] [rbp-F0h]
  unsigned int v98; // [rsp+78h] [rbp-E8h]
  __m128i *v99; // [rsp+80h] [rbp-E0h]
  size_t v100; // [rsp+88h] [rbp-D8h]
  __m128i v101; // [rsp+90h] [rbp-D0h] BYREF
  char *v102; // [rsp+A0h] [rbp-C0h] BYREF
  size_t v103; // [rsp+A8h] [rbp-B8h]
  _QWORD v104[2]; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned __int64 v105[2]; // [rsp+C0h] [rbp-A0h] BYREF
  _DWORD v106[4]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v107[2]; // [rsp+E0h] [rbp-80h] BYREF
  _BYTE v108[112]; // [rsp+F0h] [rbp-70h] BYREF

  v7 = *(_QWORD *)a1;
  v107[0] = (__int64)v108;
  v107[1] = 0x800000000LL;
  sub_2E56990((__int64)v107, v7, a3, a4, a5, a6);
  v12 = sub_2E508D0(v107, *(_QWORD *)(a1 + 8), v8, v9, v10, v11);
  if ( (_BYTE)v12 )
    goto LABEL_2;
  v14 = *(_QWORD *)(a1 + 120);
  ++*(_QWORD *)(a1 + 112);
  v92 = a1 + 112;
  v15 = *(_DWORD *)(a1 + 136);
  v96 = (_QWORD *)v14;
  v16 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  v98 = v15;
  v95 = 1;
  v97 = v16;
  sub_2E592C0((__int64 *)a1);
  v90 = 1;
  v17 = *(_QWORD *)(*(_QWORD *)a1 + 328LL);
  v93 = *(_QWORD *)a1 + 320LL;
  if ( v17 == v93 )
    goto LABEL_49;
  while ( 1 )
  {
    v94 = v17;
    v18 = *(_QWORD *)(a1 + 16);
    if ( v17 )
    {
      v19 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
      v20 = *(_DWORD *)(v17 + 24) + 1;
    }
    else
    {
      v19 = 0;
      v20 = 0;
    }
    if ( v20 < *(_DWORD *)(v18 + 32) && *(_QWORD *)(*(_QWORD *)(v18 + 24) + 8 * v19) )
    {
      v21 = *sub_2E57C80(v92, &v94);
      v22 = sub_2E57C80((__int64)&v95, &v94);
      v23 = *v22;
      if ( *(_DWORD *)(*v22 + 8) != *(_DWORD *)(v21 + 8) )
        break;
      if ( *(_DWORD *)(v23 + 12) != *(_DWORD *)(v21 + 12) )
        break;
      if ( *(_DWORD *)(v23 + 88) != *(_DWORD *)(v21 + 88) )
        break;
      v67 = 8LL * *(unsigned int *)(v23 + 32);
      if ( v67 )
      {
        v84 = *v22;
        v68 = memcmp(*(const void **)(v23 + 24), *(const void **)(v21 + 24), v67);
        v23 = v84;
        if ( v68 )
          break;
      }
      if ( *(_DWORD *)(v23 + 160) != *(_DWORD *)(v21 + 160) )
        break;
      v69 = 8LL * *(unsigned int *)(v23 + 104);
      if ( v69 )
      {
        if ( memcmp(*(const void **)(v23 + 96), *(const void **)(v21 + 96), v69) )
          break;
      }
    }
LABEL_48:
    v17 = *(_QWORD *)(v17 + 8);
    if ( v93 == v17 )
      goto LABEL_49;
  }
  if ( a2 )
  {
    v24 = sub_CB72A0();
    v25 = (_DWORD *)v24[4];
    v26 = (__int64)v24;
    if ( v24[3] - (_QWORD)v25 <= 3u )
    {
      v26 = sub_CB6200((__int64)v24, (unsigned __int8 *)"BB: ", 4u);
    }
    else
    {
      *v25 = 540688962;
      v24[4] += 4LL;
    }
    v27 = *(int *)(v94 + 24) < 0;
    v28 = abs32(*(_DWORD *)(v94 + 24));
    if ( v28 > 9 )
    {
      if ( v28 <= 0x63 )
      {
        v74 = *(int *)(v94 + 24) < 0;
        v86 = v28;
        v78 = v26;
        v102 = (char *)v104;
        sub_2240A50((__int64 *)&v102, (unsigned int)(v27 + 2), 45);
        v35 = v78;
        v36 = v86;
        v38 = &v102[v74];
      }
      else
      {
        if ( v28 <= 0x3E7 )
        {
          v34 = 2;
          v31 = 3;
          v29 = v28;
        }
        else
        {
          v29 = v28;
          v30 = v28;
          if ( v28 <= 0x270F )
          {
            v34 = 3;
            v31 = 4;
          }
          else
          {
            v31 = 1;
            do
            {
              v32 = v30;
              v33 = v31;
              v31 += 4;
              v30 /= 0x2710u;
              if ( v32 <= 0x1869F )
              {
                v34 = v33 + 3;
                goto LABEL_24;
              }
              if ( (unsigned int)v30 <= 0x63 )
              {
                v72 = v31;
                v75 = v26;
                v87 = v28;
                v79 = *(int *)(v94 + 24) < 0;
                v102 = (char *)v104;
                sub_2240A50((__int64 *)&v102, (unsigned int)(v33 + v27 + 5), 45);
                v36 = v87;
                v35 = v75;
                v38 = &v102[v79];
                v37 = v72;
                goto LABEL_25;
              }
              if ( (unsigned int)v30 <= 0x3E7 )
              {
                v31 = v33 + 6;
                v34 = v33 + 5;
                goto LABEL_24;
              }
            }
            while ( (unsigned int)v30 > 0x270F );
            v31 = v33 + 7;
            v34 = v33 + 6;
          }
        }
LABEL_24:
        v71 = *(int *)(v94 + 24) < 0;
        v73 = v34;
        v85 = v28;
        v77 = v26;
        v102 = (char *)v104;
        sub_2240A50((__int64 *)&v102, (unsigned int)(v31 + v27), 45);
        v35 = v77;
        v36 = v85;
        v37 = v73;
        v38 = &v102[v71];
LABEL_25:
        for ( i = v29; ; i = v36 )
        {
          v40 = (unsigned __int64)(1374389535 * i) >> 37;
          v41 = v36 - 100 * v40;
          v42 = v36;
          v36 = v40;
          v43 = (unsigned int)(2 * v41);
          v44 = (unsigned int)(v43 + 1);
          LOBYTE(v43) = a00010203040506[v43];
          v38[v37] = a00010203040506[v44];
          v45 = (unsigned int)(v37 - 1);
          v37 -= 2;
          v38[v45] = v43;
          if ( v42 <= 0x270F )
            break;
        }
        if ( v42 <= 0x3E7 )
          goto LABEL_29;
      }
      v70 = 2 * v36;
      v38[1] = a00010203040506[(unsigned int)(v70 + 1)];
      *v38 = a00010203040506[v70];
      goto LABEL_30;
    }
    v76 = *(int *)(v94 + 24) < 0;
    v88 = v28;
    v81 = v26;
    v102 = (char *)v104;
    sub_2240A50((__int64 *)&v102, (unsigned int)(v27 + 1), 45);
    v35 = v81;
    LOBYTE(v36) = v88;
    v38 = &v102[v76];
LABEL_29:
    *v38 = v36 + 48;
LABEL_30:
    v106[0] = 3039842;
    v105[0] = (unsigned __int64)v106;
    v105[1] = 3;
    if ( v103 + 3 <= 0xF || v102 == (char *)v104 || v103 + 3 > v104[0] )
    {
      v82 = v35;
      v46 = sub_2241490(v105, v102, v103);
      v99 = &v101;
      v47 = v82;
      v48 = (unsigned __int8 *)*v46;
      v49 = v46 + 2;
      if ( (unsigned __int64 *)*v46 != v46 + 2 )
      {
LABEL_34:
        v99 = (__m128i *)v48;
        v101.m128i_i64[0] = v46[2];
LABEL_35:
        v100 = v46[1];
        *v46 = (unsigned __int64)v49;
        v46[1] = 0;
        *((_BYTE *)v46 + 16) = 0;
        if ( (_DWORD *)v105[0] != v106 )
        {
          v83 = v47;
          j_j___libc_free_0(v105[0]);
          v47 = v83;
        }
        if ( v102 != (char *)v104 )
        {
          v91 = v47;
          j_j___libc_free_0((unsigned __int64)v102);
          v47 = v91;
        }
        v50 = sub_CB6200(v47, (unsigned __int8 *)v99, v100);
        v51 = *(_BYTE **)(v50 + 32);
        if ( *(_BYTE **)(v50 + 24) == v51 )
        {
          sub_CB6200(v50, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v51 = 10;
          ++*(_QWORD *)(v50 + 32);
        }
        if ( v99 != &v101 )
          j_j___libc_free_0((unsigned __int64)v99);
        v52 = *sub_2E57C80(v92, &v94);
        v53 = sub_CB72A0();
        v54 = (__m128i *)*((_QWORD *)v53 + 4);
        v55 = (__int64)v53;
        if ( *((_QWORD *)v53 + 3) - (_QWORD)v54 <= 0xFu )
        {
          v55 = sub_CB6200((__int64)v53, "Correct RP Info\n", 0x10u);
        }
        else
        {
          *v54 = _mm_load_si128((const __m128i *)&xmmword_42EAB10);
          *((_QWORD *)v53 + 4) += 16LL;
        }
        sub_2E50B90(a1, v55, v52);
        v56 = *sub_2E57C80((__int64)&v95, &v94);
        v57 = sub_CB72A0();
        v58 = (__m128i *)v57[4];
        v59 = (__int64)v57;
        if ( v57[3] - (_QWORD)v58 <= 0x11u )
        {
          v59 = sub_CB6200((__int64)v57, "Incorrect RP Info\n", 0x12u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42EAB20);
          v58[1].m128i_i16[0] = 2671;
          *v58 = si128;
          v57[4] += 18LL;
        }
        sub_2E50B90(a1, v59, v56);
        v90 = 0;
        goto LABEL_48;
      }
    }
    else
    {
      v80 = v35;
      v46 = sub_2241130((unsigned __int64 *)&v102, 0, 0, v106, 3u);
      v99 = &v101;
      v48 = (unsigned __int8 *)*v46;
      v49 = v46 + 2;
      v47 = v80;
      if ( (unsigned __int64 *)*v46 != v46 + 2 )
        goto LABEL_34;
    }
    v101 = _mm_loadu_si128((const __m128i *)v46 + 1);
    goto LABEL_35;
  }
  v90 = 0;
LABEL_49:
  v61 = v98;
  if ( v98 )
  {
    v62 = v96;
    v63 = &v96[2 * v98];
    do
    {
      if ( *v62 != -4096 && *v62 != -8192 )
      {
        v64 = v62[1];
        if ( v64 )
        {
          v65 = *(_QWORD *)(v64 + 96);
          if ( v65 != v64 + 112 )
            _libc_free(v65);
          v66 = *(_QWORD *)(v64 + 24);
          if ( v66 != v64 + 40 )
            _libc_free(v66);
          j_j___libc_free_0(v64);
        }
      }
      v62 += 2;
    }
    while ( v63 != v62 );
    v61 = v98;
  }
  sub_C7D6A0((__int64)v96, 16LL * v61, 8);
  v12 = v90;
LABEL_2:
  if ( (_BYTE *)v107[0] != v108 )
    _libc_free(v107[0]);
  return v12;
}
