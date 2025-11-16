// Function: sub_26B2BA0
// Address: 0x26b2ba0
//
__int64 __fastcall sub_26B2BA0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  unsigned __int64 v10; // r14
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r13
  _QWORD *v13; // rax
  __int64 v14; // rcx
  _QWORD *v15; // rdx
  char v16; // cl
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 *v20; // r14
  __int64 v21; // r12
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  _QWORD *v25; // r10
  unsigned int v26; // ecx
  _QWORD *v27; // r15
  __int64 v28; // rsi
  int v29; // ecx
  __int64 v30; // rdx
  unsigned __int64 *v31; // rdi
  __int64 *v32; // rdx
  _QWORD *v33; // r15
  __int64 v34; // rax
  __int64 v35; // r15
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  unsigned int v38; // ecx
  _QWORD *v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // r13
  _QWORD *v42; // rax
  __int64 v43; // r15
  __int64 v44; // r12
  int v45; // r14d
  __int64 *v46; // r9
  __int64 v47; // r8
  __int64 v48; // r13
  __int64 v49; // rdx
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rsi
  int v52; // eax
  __int64 v53; // rcx
  __int64 **v54; // r14
  __int64 v55; // rdx
  _QWORD *v56; // r12
  unsigned int v57; // eax
  int v59; // r10d
  int v60; // r10d
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 *v63; // rdi
  __int64 v64; // r12
  _QWORD *v65; // rbx
  _QWORD *v66; // r12
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rax
  unsigned int v70; // eax
  _QWORD *v71; // rbx
  _QWORD *v72; // r12
  __int64 v73; // rsi
  __int64 *src; // [rsp+10h] [rbp-190h]
  __int64 v76; // [rsp+18h] [rbp-188h]
  size_t n; // [rsp+30h] [rbp-170h]
  __int64 v79; // [rsp+48h] [rbp-158h]
  __int64 v80; // [rsp+60h] [rbp-140h] BYREF
  _QWORD *v81; // [rsp+68h] [rbp-138h]
  __int64 v82; // [rsp+70h] [rbp-130h]
  unsigned int v83; // [rsp+78h] [rbp-128h]
  _QWORD *v84; // [rsp+88h] [rbp-118h]
  unsigned int v85; // [rsp+98h] [rbp-108h]
  char v86; // [rsp+A0h] [rbp-100h]
  __int64 *v87; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v88; // [rsp+B8h] [rbp-E8h] BYREF
  __int64 v89; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v90; // [rsp+C8h] [rbp-D8h]
  __int64 v91; // [rsp+D0h] [rbp-D0h]
  __int64 *v92; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v93; // [rsp+108h] [rbp-98h] BYREF
  __int64 v94; // [rsp+110h] [rbp-90h] BYREF
  __int64 v95; // [rsp+118h] [rbp-88h]
  __int64 *i; // [rsp+120h] [rbp-80h]
  __int64 v97; // [rsp+150h] [rbp-50h]
  __int64 v98; // [rsp+158h] [rbp-48h]
  __int64 v99; // [rsp+160h] [rbp-40h]

  *(_QWORD *)a1 = a2;
  v8 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_BYTE *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = a4;
  *(_QWORD *)(a1 + 152) = a5;
  *(_QWORD *)(a1 + 160) = a6;
  *(__m128i *)(a1 + 168) = v8;
  v9 = (_QWORD *)sub_22077B0(0x1B0u);
  if ( v9 )
  {
    v9[1] = 0x400000000LL;
    *v9 = v9 + 2;
  }
  v10 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 128) = v9;
  if ( v10 )
  {
    v11 = *(unsigned __int64 **)v10;
    v12 = (unsigned __int64 *)(*(_QWORD *)v10 + 104LL * *(unsigned int *)(v10 + 8));
    if ( *(unsigned __int64 **)v10 != v12 )
    {
      do
      {
        v12 -= 13;
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          _libc_free(*v12);
      }
      while ( v11 != v12 );
      v12 = *(unsigned __int64 **)v10;
    }
    if ( v12 != (unsigned __int64 *)(v10 + 16) )
      _libc_free((unsigned __int64)v12);
    j_j___libc_free_0(v10);
  }
  v80 = 0;
  v83 = 128;
  v13 = (_QWORD *)sub_C7D670(0x2000, 8);
  v82 = 0;
  v81 = v13;
  v93 = 2;
  v15 = v13 + 1024;
  v92 = (__int64 *)&unk_49DD7B0;
  v94 = 0;
  v95 = -4096;
  for ( i = 0; v15 != v13; v13 += 8 )
  {
    if ( v13 )
    {
      v16 = v93;
      v13[2] = 0;
      v13[3] = -4096;
      *v13 = &unk_49DD7B0;
      v13[1] = v16 & 6;
      v14 = (__int64)i;
      v13[4] = i;
    }
  }
  v86 = 0;
  v17 = sub_F4BFF0(a2, (__int64)&v80, 0, v14);
  v18 = *a3;
  v19 = v17;
  *(_QWORD *)(a1 + 8) = v17;
  v76 = v18 + 104LL * *((unsigned int *)a3 + 2);
  if ( v18 != v76 )
  {
    v79 = v18;
    while ( 1 )
    {
      v87 = &v89;
      v88 = 0x800000000LL;
      v20 = *(__int64 **)v79;
      v21 = *(_QWORD *)v79 + 8LL * *(unsigned int *)(v79 + 8);
      if ( v21 != *(_QWORD *)v79 )
      {
        while ( 1 )
        {
          v37 = *v20;
          v93 = 2;
          v94 = 0;
          v95 = v37;
          if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
            sub_BD73F0((__int64)&v93);
          i = &v80;
          v92 = (__int64 *)&unk_49DD7B0;
          if ( !v83 )
            break;
          v24 = v95;
          v23 = v83 - 1;
          v22 = (__int64)v81;
          v38 = v23 & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
          v39 = &v81[8 * (unsigned __int64)v38];
          v40 = v39[3];
          if ( v40 != v95 )
          {
            v59 = 1;
            v27 = 0;
            while ( v40 != -4096 )
            {
              if ( v40 == -8192 && !v27 )
                v27 = v39;
              v38 = v23 & (v59 + v38);
              v39 = &v81[8 * (unsigned __int64)v38];
              v40 = v39[3];
              if ( v95 == v40 )
                goto LABEL_45;
              ++v59;
            }
            if ( !v27 )
              v27 = v39;
            ++v80;
            v29 = v82 + 1;
            if ( 4 * ((int)v82 + 1) < 3 * v83 )
            {
              if ( v83 - HIDWORD(v82) - v29 <= v83 >> 3 )
              {
                sub_CF32C0((__int64)&v80, v83);
                if ( v83 )
                {
                  v24 = v95;
                  v22 = 0;
                  v60 = 1;
                  LODWORD(v61) = (v83 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
                  v27 = &v81[8 * (unsigned __int64)(unsigned int)v61];
                  v62 = v27[3];
                  if ( v62 != v95 )
                  {
                    while ( v62 != -4096 )
                    {
                      if ( v62 == -8192 && !v22 )
                        v22 = (__int64)v27;
                      v23 = (unsigned int)(v60 + 1);
                      v61 = (v83 - 1) & ((_DWORD)v61 + v60);
                      v27 = &v81[8 * v61];
                      v62 = v27[3];
                      if ( v95 == v62 )
                        goto LABEL_23;
                      ++v60;
                    }
                    if ( v22 )
                      v27 = (_QWORD *)v22;
                  }
                  goto LABEL_23;
                }
LABEL_79:
                v24 = v95;
                v27 = 0;
                goto LABEL_23;
              }
LABEL_24:
              v30 = v27[3];
              LODWORD(v82) = v29;
              if ( v30 == -4096 )
              {
                v31 = v27 + 1;
                if ( v24 != -4096 )
                  goto LABEL_29;
              }
              else
              {
                --HIDWORD(v82);
                if ( v30 != v24 )
                {
                  v31 = v27 + 1;
                  if ( v30 != -8192 && v30 )
                  {
                    sub_BD60C0(v31);
                    v24 = v95;
                    v31 = v27 + 1;
                  }
LABEL_29:
                  v27[3] = v24;
                  if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
                    sub_BD6050(v31, v93 & 0xFFFFFFFFFFFFFFF8LL);
                  v24 = v95;
                }
              }
              v32 = i;
              v33 = v27 + 5;
              *v33 = 6;
              v33[1] = 0;
              *(v33 - 1) = v32;
              v33[2] = 0;
              goto LABEL_34;
            }
LABEL_21:
            sub_CF32C0((__int64)&v80, 2 * v83);
            if ( !v83 )
              goto LABEL_79;
            v24 = v95;
            v25 = 0;
            v22 = 1;
            v26 = (v83 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
            v27 = &v81[8 * (unsigned __int64)v26];
            v28 = v27[3];
            if ( v28 != v95 )
            {
              while ( v28 != -4096 )
              {
                if ( !v25 && v28 == -8192 )
                  v25 = v27;
                v23 = (unsigned int)(v22 + 1);
                v26 = (v83 - 1) & (v22 + v26);
                v27 = &v81[8 * (unsigned __int64)v26];
                v28 = v27[3];
                if ( v95 == v28 )
                  goto LABEL_23;
                v22 = (unsigned int)v23;
              }
              if ( v25 )
                v27 = v25;
            }
LABEL_23:
            v29 = v82 + 1;
            goto LABEL_24;
          }
LABEL_45:
          v33 = v39 + 5;
LABEL_34:
          v92 = (__int64 *)&unk_49DB368;
          if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
            sub_BD60C0(&v93);
          v34 = (unsigned int)v88;
          v35 = v33[2];
          v36 = (unsigned int)v88 + 1LL;
          if ( v36 > HIDWORD(v88) )
          {
            sub_C8D5F0((__int64)&v87, &v89, v36, 8u, v22, v23);
            v34 = (unsigned int)v88;
          }
          ++v20;
          v87[v34] = v35;
          LODWORD(v88) = v88 + 1;
          if ( (__int64 *)v21 == v20 )
            goto LABEL_48;
        }
        ++v80;
        goto LABEL_21;
      }
LABEL_48:
      v41 = sub_26B2930((__int64)&v80, *(_QWORD *)(v79 + 80))[2];
      v42 = sub_26B2930((__int64)&v80, *(_QWORD *)(v79 + 88));
      v43 = *(_QWORD *)(v79 + 96);
      v44 = v42[2];
      if ( v43 )
        v43 = sub_26B2930((__int64)&v80, *(_QWORD *)(v79 + 96))[2];
      v45 = v88;
      v46 = v87;
      v92 = &v94;
      v47 = 8LL * (unsigned int)v88;
      v93 = 0x800000000LL;
      if ( (unsigned int)v88 > 8uLL )
        break;
      if ( v47 )
      {
        v63 = &v94;
        goto LABEL_81;
      }
LABEL_52:
      v97 = v41;
      LODWORD(v93) = v47 + v45;
      v48 = *(_QWORD *)(a1 + 128);
      v98 = v44;
      v99 = v43;
      v49 = *(unsigned int *)(v48 + 8);
      v50 = *(unsigned int *)(v48 + 12);
      v51 = v49 + 1;
      v52 = *(_DWORD *)(v48 + 8);
      if ( v49 + 1 > v50 )
      {
        v64 = *(_QWORD *)v48;
        if ( *(_QWORD *)v48 > (unsigned __int64)&v92 || (unsigned __int64)&v92 >= v64 + 104 * v49 )
        {
          sub_26AB4F0(v48, v51, v49, v50, v47, (__int64)v46);
          v49 = *(unsigned int *)(v48 + 8);
          v53 = *(_QWORD *)v48;
          v54 = &v92;
          v52 = *(_DWORD *)(v48 + 8);
        }
        else
        {
          sub_26AB4F0(v48, v51, v49, v50, v47, (__int64)v46);
          v53 = *(_QWORD *)v48;
          v49 = *(unsigned int *)(v48 + 8);
          v54 = (__int64 **)((char *)&v92 + *(_QWORD *)v48 - v64);
          v52 = *(_DWORD *)(v48 + 8);
        }
      }
      else
      {
        v53 = *(_QWORD *)v48;
        v54 = &v92;
      }
      v55 = 13 * v49;
      v56 = (_QWORD *)(v53 + 8 * v55);
      if ( v56 )
      {
        *v56 = v56 + 2;
        v56[1] = 0x800000000LL;
        if ( *((_DWORD *)v54 + 2) )
          sub_26AB6C0(v53 + 8 * v55, (__int64)v54, v55, v53, v47, (__int64)v46);
        v56[10] = v54[10];
        v56[11] = v54[11];
        v56[12] = v54[12];
        v52 = *(_DWORD *)(v48 + 8);
      }
      *(_DWORD *)(v48 + 8) = v52 + 1;
      if ( v92 != &v94 )
        _libc_free((unsigned __int64)v92);
      if ( v87 != &v89 )
        _libc_free((unsigned __int64)v87);
      v79 += 104;
      if ( v76 == v79 )
      {
        v19 = *(_QWORD *)(a1 + 8);
        goto LABEL_64;
      }
    }
    src = v87;
    n = 8LL * (unsigned int)v88;
    sub_C8D5F0((__int64)&v92, &v94, (unsigned int)v88, 8u, v47, (__int64)v87);
    v47 = n;
    v46 = src;
    v63 = &v92[(unsigned int)v93];
LABEL_81:
    memcpy(v63, v46, v47);
    v47 = (unsigned int)v93;
    goto LABEL_52;
  }
LABEL_64:
  sub_BD84D0(a2, v19);
  if ( v86 )
  {
    v70 = v85;
    v86 = 0;
    if ( v85 )
    {
      v71 = v84;
      v72 = &v84[2 * v85];
      do
      {
        if ( *v71 != -8192 && *v71 != -4096 )
        {
          v73 = v71[1];
          if ( v73 )
            sub_B91220((__int64)(v71 + 1), v73);
        }
        v71 += 2;
      }
      while ( v72 != v71 );
      v70 = v85;
    }
    sub_C7D6A0((__int64)v84, 16LL * v70, 8);
  }
  v57 = v83;
  if ( v83 )
  {
    v65 = v81;
    v88 = 2;
    v89 = 0;
    v66 = &v81[8 * (unsigned __int64)v83];
    v90 = -4096;
    v67 = -4096;
    v87 = (__int64 *)&unk_49DD7B0;
    v91 = 0;
    v93 = 2;
    v94 = 0;
    v95 = -8192;
    v92 = (__int64 *)&unk_49DD7B0;
    i = 0;
    while ( 1 )
    {
      v68 = v65[3];
      if ( v67 != v68 )
      {
        v67 = v95;
        if ( v68 != v95 )
        {
          v69 = v65[7];
          if ( v69 != -4096 && v69 != 0 && v69 != -8192 )
          {
            sub_BD60C0(v65 + 5);
            v68 = v65[3];
          }
          v67 = v68;
        }
      }
      *v65 = &unk_49DB368;
      if ( v67 != -4096 && v67 != 0 && v67 != -8192 )
        sub_BD60C0(v65 + 1);
      v65 += 8;
      if ( v66 == v65 )
        break;
      v67 = v90;
    }
    v92 = (__int64 *)&unk_49DB368;
    if ( v95 != -4096 && v95 != 0 && v95 != -8192 )
      sub_BD60C0(&v93);
    v87 = (__int64 *)&unk_49DB368;
    if ( v90 != -4096 && v90 != 0 && v90 != -8192 )
      sub_BD60C0(&v88);
    v57 = v83;
  }
  return sub_C7D6A0((__int64)v81, (unsigned __int64)v57 << 6, 8);
}
