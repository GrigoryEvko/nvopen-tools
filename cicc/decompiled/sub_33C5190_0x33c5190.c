// Function: sub_33C5190
// Address: 0x33c5190
//
void __fastcall sub_33C5190(__int64 a1, __int64 a2, __int64 *a3, _BYTE *a4, __int64 a5)
{
  __int64 v8; // r15
  __int64 v9; // r10
  __int64 v10; // rax
  __int64 *v11; // r11
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // r9
  __int64 *v15; // r11
  __int64 v16; // r10
  const __m128i *v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  __m128i *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r9
  const __m128i *v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  __m128i *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rax
  int v40; // r8d
  int v41; // ecx
  __int64 v42; // rax
  _QWORD *v43; // rbx
  __int64 v44; // r12
  __int64 v45; // rsi
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // r15
  __int64 v49; // rsi
  unsigned int v50; // eax
  bool v51; // al
  bool v52; // r15
  __int64 v53; // rax
  unsigned int v54; // eax
  bool v55; // al
  bool v56; // r14
  __int64 v57; // r15
  const void *v58; // rsi
  const void *v59; // rsi
  unsigned __int64 v60; // [rsp+0h] [rbp-100h]
  unsigned __int64 v61; // [rsp+8h] [rbp-F8h]
  __int64 *v62; // [rsp+8h] [rbp-F8h]
  __int64 *v63; // [rsp+8h] [rbp-F8h]
  __int64 *v64; // [rsp+10h] [rbp-F0h]
  __int64 *v65; // [rsp+10h] [rbp-F0h]
  __int64 *v66; // [rsp+10h] [rbp-F0h]
  __int64 *v67; // [rsp+10h] [rbp-F0h]
  __int64 *v68; // [rsp+10h] [rbp-F0h]
  __int64 v69; // [rsp+10h] [rbp-F0h]
  __int64 v70; // [rsp+10h] [rbp-F0h]
  __int64 v71; // [rsp+18h] [rbp-E8h]
  __int64 v72; // [rsp+18h] [rbp-E8h]
  _BYTE *v73; // [rsp+18h] [rbp-E8h]
  __int64 v74; // [rsp+18h] [rbp-E8h]
  __int64 v75; // [rsp+18h] [rbp-E8h]
  __int64 v76; // [rsp+18h] [rbp-E8h]
  __int64 v78; // [rsp+28h] [rbp-D8h]
  int v80; // [rsp+30h] [rbp-D0h]
  __int64 v81; // [rsp+38h] [rbp-C8h]
  int v82; // [rsp+38h] [rbp-C8h]
  __int64 v83; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v84; // [rsp+48h] [rbp-B8h]
  __int64 v85; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+58h] [rbp-A8h]
  int v87; // [rsp+60h] [rbp-A0h]
  int v88; // [rsp+64h] [rbp-9Ch]
  __int64 v89; // [rsp+70h] [rbp-90h] BYREF
  _BYTE *v90; // [rsp+78h] [rbp-88h]
  __int64 v91; // [rsp+80h] [rbp-80h]
  __int64 v92; // [rsp+88h] [rbp-78h]
  __int64 v93; // [rsp+90h] [rbp-70h]
  __int64 v94; // [rsp+98h] [rbp-68h]
  __int64 v95; // [rsp+A0h] [rbp-60h]
  __int64 v96; // [rsp+A8h] [rbp-58h] BYREF
  unsigned int v97; // [rsp+B0h] [rbp-50h]
  __int64 v98; // [rsp+B8h] [rbp-48h] BYREF
  int v99; // [rsp+C0h] [rbp-40h]
  int v100; // [rsp+C4h] [rbp-3Ch]
  char v101; // [rsp+C8h] [rbp-38h]

  sub_35D8B00(&v85, *(_QWORD *)(a1 + 896));
  v8 = a3[1];
  v9 = v86;
  v78 = a3[2];
  v81 = *(_QWORD *)(v86 + 8);
  v10 = *a3;
  v11 = *(__int64 **)(*a3 + 8);
  if ( v85 == v8 && !*(_DWORD *)v8 && *(_QWORD *)(v8 + 8) == a3[3] )
  {
    v53 = *(_QWORD *)(v8 + 16);
    v84 = *(_DWORD *)(v53 + 32);
    if ( v84 > 0x40 )
    {
      v63 = v11;
      v70 = v86;
      sub_C43780((__int64)&v83, (const void **)(v53 + 24));
      v11 = v63;
      v9 = v70;
    }
    else
    {
      v83 = *(_QWORD *)(v53 + 24);
    }
    v68 = v11;
    v75 = v9;
    sub_C46A40((__int64)&v83, 1);
    v54 = v84;
    v84 = 0;
    v9 = v75;
    v11 = v68;
    LODWORD(v90) = v54;
    v89 = v83;
    if ( v54 <= 0x40 )
    {
      if ( v83 == *(_QWORD *)(v81 + 24) )
      {
LABEL_53:
        v13 = *(_QWORD *)(v8 + 24);
        goto LABEL_5;
      }
    }
    else
    {
      v61 = v83;
      v55 = sub_C43C50((__int64)&v89, (const void **)(v81 + 24));
      v9 = v75;
      v11 = v68;
      v56 = v55;
      if ( v61 )
      {
        j_j___libc_free_0_0(v61);
        v9 = v75;
        v11 = v68;
        if ( v84 > 0x40 )
        {
          if ( v83 )
          {
            j_j___libc_free_0_0(v83);
            v11 = v68;
            v9 = v75;
          }
        }
      }
      if ( v56 )
        goto LABEL_53;
    }
    v10 = *a3;
  }
  v64 = v11;
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL);
  LOBYTE(v90) = 0;
  v71 = v9;
  v13 = sub_2E7AAE0(v12, *(_QWORD *)(v10 + 16), v89, 0);
  sub_2E33BD0(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL) + 320LL, v13);
  v15 = v64;
  v16 = v71;
  v17 = (const __m128i *)&v89;
  v18 = *v64;
  v19 = *(_QWORD *)v13 & 7LL;
  *(_QWORD *)(v13 + 8) = v64;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v13 = v18 | v19;
  *(_QWORD *)(v18 + 8) = v13;
  *v64 = v13 | *v64 & 7;
  v20 = *(unsigned int *)(a2 + 12);
  v89 = v13;
  v91 = v85;
  v21 = a3[3];
  v90 = (_BYTE *)v8;
  v92 = v21;
  v93 = v81;
  LODWORD(v94) = *((_DWORD *)a3 + 10) >> 1;
  v22 = *(unsigned int *)(a2 + 8);
  v23 = v22 + 1;
  if ( v22 + 1 > v20 )
  {
    v57 = *(_QWORD *)a2;
    v58 = (const void *)(a2 + 16);
    if ( *(_QWORD *)a2 > (unsigned __int64)&v89 || (unsigned __int64)&v89 >= v57 + 48 * v22 )
    {
      sub_C8D5F0(a2, v58, v23, 0x30u, (__int64)&v89, v14);
      v24 = *(_QWORD *)a2;
      v22 = *(unsigned int *)(a2 + 8);
      v17 = (const __m128i *)&v89;
      v15 = v64;
      v16 = v71;
    }
    else
    {
      sub_C8D5F0(a2, v58, v23, 0x30u, (__int64)&v89, v14);
      v24 = *(_QWORD *)a2;
      v22 = *(unsigned int *)(a2 + 8);
      v16 = v71;
      v15 = v64;
      v17 = (const __m128i *)((char *)&v89 + *(_QWORD *)a2 - v57);
    }
  }
  else
  {
    v24 = *(_QWORD *)a2;
  }
  v65 = v15;
  v25 = (__m128i *)(v24 + 48 * v22);
  v72 = v16;
  *v25 = _mm_loadu_si128(v17);
  v25[1] = _mm_loadu_si128(v17 + 1);
  v25[2] = _mm_loadu_si128(v17 + 2);
  ++*(_DWORD *)(a2 + 8);
  sub_33C4170(a1, a4);
  v11 = v65;
  v9 = v72;
LABEL_5:
  if ( v78 == v9 && !*(_DWORD *)v78 )
  {
    v48 = a3[4];
    if ( v48 )
    {
      v49 = *(_QWORD *)(v78 + 16);
      v84 = *(_DWORD *)(v49 + 32);
      if ( v84 > 0x40 )
      {
        v62 = v11;
        v69 = v9;
        sub_C43780((__int64)&v83, (const void **)(v49 + 24));
        v11 = v62;
        v9 = v69;
      }
      else
      {
        v83 = *(_QWORD *)(v49 + 24);
      }
      v67 = v11;
      v74 = v9;
      sub_C46A40((__int64)&v83, 1);
      v50 = v84;
      v84 = 0;
      v9 = v74;
      v11 = v67;
      LODWORD(v90) = v50;
      v89 = v83;
      if ( v50 <= 0x40 )
      {
        if ( v83 == *(_QWORD *)(v48 + 24) )
        {
LABEL_42:
          v28 = *(_QWORD *)(v9 + 24);
          goto LABEL_9;
        }
      }
      else
      {
        v60 = v83;
        v51 = sub_C43C50((__int64)&v89, (const void **)(v48 + 24));
        v9 = v74;
        v11 = v67;
        v52 = v51;
        if ( v60 )
        {
          j_j___libc_free_0_0(v60);
          v9 = v74;
          v11 = v67;
          if ( v84 > 0x40 )
          {
            if ( v83 )
            {
              j_j___libc_free_0_0(v83);
              v11 = v67;
              v9 = v74;
            }
          }
        }
        if ( v52 )
          goto LABEL_42;
      }
    }
  }
  v66 = v11;
  v26 = *(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL);
  v27 = *a3;
  LOBYTE(v90) = 0;
  v73 = (_BYTE *)v9;
  v28 = sub_2E7AAE0(v26, *(_QWORD *)(v27 + 16), v89, 0);
  sub_2E33BD0(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL) + 320LL, v28);
  v30 = (const __m128i *)&v89;
  v31 = *v66;
  v32 = *(_QWORD *)v28 & 7LL;
  *(_QWORD *)(v28 + 8) = v66;
  v31 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v28 = v31 | v32;
  *(_QWORD *)(v31 + 8) = v28;
  *v66 = v28 | *v66 & 7;
  v33 = *(unsigned int *)(a2 + 8);
  v89 = v28;
  v91 = v78;
  v34 = v33 + 1;
  v90 = v73;
  v92 = v81;
  v93 = a3[4];
  LODWORD(v94) = *((_DWORD *)a3 + 10) >> 1;
  if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    v59 = (const void *)(a2 + 16);
    if ( *(_QWORD *)a2 > (unsigned __int64)&v89
      || (v76 = *(_QWORD *)a2, (unsigned __int64)&v89 >= *(_QWORD *)a2 + 48 * v33) )
    {
      sub_C8D5F0(a2, v59, v34, 0x30u, (__int64)&v89, v29);
      v35 = *(_QWORD *)a2;
      v33 = *(unsigned int *)(a2 + 8);
      v30 = (const __m128i *)&v89;
    }
    else
    {
      sub_C8D5F0(a2, v59, v34, 0x30u, (__int64)&v89, v29);
      v35 = *(_QWORD *)a2;
      v33 = *(unsigned int *)(a2 + 8);
      v30 = (const __m128i *)((char *)&v89 + *(_QWORD *)a2 - v76);
    }
  }
  else
  {
    v35 = *(_QWORD *)a2;
  }
  v36 = (__m128i *)(v35 + 48 * v33);
  *v36 = _mm_loadu_si128(v30);
  v36[1] = _mm_loadu_si128(v30 + 1);
  v36[2] = _mm_loadu_si128(v30 + 2);
  ++*(_DWORD *)(a2 + 8);
  sub_33C4170(a1, a4);
LABEL_9:
  v37 = *(_QWORD *)a1;
  v84 = *(_DWORD *)(a1 + 848);
  if ( v37 && &v83 != (__int64 *)(v37 + 48) && (v38 = *(_QWORD *)(v37 + 48), (v83 = v38) != 0) )
  {
    sub_B96E90((__int64)&v83, v38, 1);
    v39 = *a3;
    LODWORD(v89) = 20;
    v40 = v87;
    v93 = v13;
    v90 = a4;
    v91 = 0;
    v41 = v88;
    v92 = v81;
    v94 = v28;
    v95 = v39;
    v96 = v83;
    if ( v83 )
    {
      v80 = v87;
      v82 = v88;
      sub_B96E90((__int64)&v96, v83, 1);
      v101 = 0;
      v98 = 0;
      v97 = v84;
      v99 = v80;
      v100 = v82;
      if ( v83 )
        sub_B91220((__int64)&v83, v83);
      v42 = a5;
      if ( *a3 != a5 )
        goto LABEL_16;
LABEL_31:
      sub_3391190(a1, (unsigned int *)&v89, v42);
      goto LABEL_24;
    }
  }
  else
  {
    v47 = *a3;
    v93 = v13;
    LODWORD(v89) = 20;
    v40 = v87;
    v90 = a4;
    v91 = 0;
    v41 = v88;
    v92 = v81;
    v94 = v28;
    v95 = v47;
    v96 = 0;
  }
  v98 = 0;
  v99 = v40;
  v97 = v84;
  v42 = a5;
  v100 = v41;
  v101 = 0;
  if ( *a3 == a5 )
    goto LABEL_31;
LABEL_16:
  v43 = *(_QWORD **)(a1 + 896);
  v44 = v43[2];
  if ( v44 == v43[3] )
  {
    sub_3376950(v43 + 1, v43[2], (__int64)&v89);
  }
  else
  {
    if ( v44 )
    {
      *(_QWORD *)v44 = v89;
      *(_QWORD *)(v44 + 8) = v90;
      *(_QWORD *)(v44 + 16) = v91;
      *(_QWORD *)(v44 + 24) = v92;
      *(_QWORD *)(v44 + 32) = v93;
      *(_QWORD *)(v44 + 40) = v94;
      *(_QWORD *)(v44 + 48) = v95;
      v45 = v96;
      *(_QWORD *)(v44 + 56) = v96;
      if ( v45 )
        sub_B96E90(v44 + 56, v45, 1);
      *(_DWORD *)(v44 + 64) = v97;
      v46 = v98;
      *(_QWORD *)(v44 + 72) = v98;
      if ( v46 )
        sub_B96E90(v44 + 72, v46, 1);
      *(_DWORD *)(v44 + 80) = v99;
      *(_DWORD *)(v44 + 84) = v100;
      *(_BYTE *)(v44 + 88) = v101;
      v44 = v43[2];
    }
    v43[2] = v44 + 96;
  }
LABEL_24:
  if ( v98 )
    sub_B91220((__int64)&v98, v98);
  if ( v96 )
    sub_B91220((__int64)&v96, v96);
}
