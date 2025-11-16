// Function: sub_26B3730
// Address: 0x26b3730
//
__int64 __fastcall sub_26B3730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rdx
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  char v15; // cl
  __int64 v16; // r8
  __int64 *v17; // r15
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // r9
  __int64 v21; // rax
  _QWORD *v22; // r14
  int v23; // ecx
  __int64 v24; // rdx
  unsigned __int64 *v25; // rdi
  __int64 *v26; // rdx
  _QWORD *v27; // r14
  unsigned int v28; // ecx
  _QWORD *v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 *v33; // r15
  __int64 *v34; // r13
  __int64 *v35; // r14
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rcx
  int v39; // edi
  __int64 v40; // rdx
  unsigned __int64 *v41; // rdi
  __int64 *v42; // rdx
  _QWORD *v43; // rdx
  unsigned int v44; // edi
  _QWORD *v45; // rdx
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // r10
  __int64 v49; // rdx
  __int64 v50; // rsi
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  _QWORD *v57; // r10
  unsigned int v58; // esi
  __int64 v59; // rdi
  unsigned int v60; // edx
  int v61; // r11d
  unsigned int v62; // ecx
  __int64 v63; // rsi
  unsigned int v64; // eax
  _QWORD *v65; // rbx
  _QWORD *v66; // r12
  __int64 v67; // rsi
  int v68; // r11d
  unsigned int v69; // esi
  __int64 v70; // rdi
  int v71; // r10d
  _QWORD *v72; // r11
  unsigned int v73; // ecx
  __int64 v74; // rsi
  unsigned int v75; // r11d
  _QWORD *v76; // r10
  _QWORD *v79; // [rsp+18h] [rbp-F8h]
  __int64 v80; // [rsp+18h] [rbp-F8h]
  __int64 v81; // [rsp+20h] [rbp-F0h]
  unsigned __int64 *v82; // [rsp+20h] [rbp-F0h]
  _QWORD *v83; // [rsp+20h] [rbp-F0h]
  _QWORD *v84; // [rsp+20h] [rbp-F0h]
  __int64 v85; // [rsp+20h] [rbp-F0h]
  _QWORD v86[2]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v87; // [rsp+48h] [rbp-C8h]
  __int64 v88; // [rsp+50h] [rbp-C0h]
  void *v89; // [rsp+60h] [rbp-B0h]
  __int64 v90; // [rsp+68h] [rbp-A8h] BYREF
  __int64 v91; // [rsp+70h] [rbp-A0h]
  __int64 v92; // [rsp+78h] [rbp-98h]
  __int64 *i; // [rsp+80h] [rbp-90h]
  __int64 v94; // [rsp+90h] [rbp-80h] BYREF
  _QWORD *v95; // [rsp+98h] [rbp-78h]
  __int64 v96; // [rsp+A0h] [rbp-70h]
  unsigned int v97; // [rsp+A8h] [rbp-68h]
  _QWORD *v98; // [rsp+B8h] [rbp-58h]
  unsigned int v99; // [rsp+C8h] [rbp-48h]
  char v100; // [rsp+D0h] [rbp-40h]

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
  v9 = (_QWORD *)sub_22077B0(0x70u);
  if ( v9 )
  {
    memset(v9, 0, 0x70u);
    v9[1] = 0x400000000LL;
    *v9 = v9 + 2;
    v9[8] = v9 + 10;
    v9[9] = 0x400000000LL;
  }
  v10 = *(unsigned __int64 **)(a1 + 120);
  *(_QWORD *)(a1 + 120) = v9;
  if ( v10 )
  {
    v11 = v10[8];
    if ( (unsigned __int64 *)v11 != v10 + 10 )
      _libc_free(v11);
    if ( (unsigned __int64 *)*v10 != v10 + 2 )
      _libc_free(*v10);
    j_j___libc_free_0((unsigned __int64)v10);
  }
  v94 = 0;
  v97 = 128;
  v12 = (_QWORD *)sub_C7D670(0x2000, 8);
  v96 = 0;
  v95 = v12;
  v90 = 2;
  v14 = v12 + 1024;
  v89 = &unk_49DD7B0;
  v91 = 0;
  v92 = -4096;
  for ( i = 0; v14 != v12; v12 += 8 )
  {
    if ( v12 )
    {
      v15 = v90;
      v12[2] = 0;
      v12[3] = -4096;
      *v12 = &unk_49DD7B0;
      v12[1] = v15 & 6;
      v13 = (__int64)i;
      v12[4] = i;
    }
  }
  v100 = 0;
  *(_QWORD *)(a1 + 8) = sub_F4BFF0(a2, (__int64)&v94, 0, v13);
  *(_QWORD *)(*(_QWORD *)(a1 + 120) + 48LL) = sub_26B2930((__int64)&v94, *(_QWORD *)(a3 + 48))[2];
  *(_QWORD *)(*(_QWORD *)(a1 + 120) + 56LL) = sub_26B2930((__int64)&v94, *(_QWORD *)(a3 + 56))[2];
  v81 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v81 )
  {
    v17 = *(__int64 **)a3;
    while ( 1 )
    {
      v18 = *v17;
      v19 = *(_QWORD *)(a1 + 120);
      v90 = 2;
      v91 = 0;
      v92 = v18;
      if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        sub_BD73F0((__int64)&v90);
      i = &v94;
      v89 = &unk_49DD7B0;
      if ( !v97 )
        break;
      v21 = v92;
      v20 = (__int64)v95;
      v28 = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
      v29 = &v95[8 * (unsigned __int64)v28];
      v30 = v29[3];
      if ( v92 != v30 )
      {
        v16 = 1;
        v22 = 0;
        while ( v30 != -4096 )
        {
          if ( !v22 && v30 == -8192 )
            v22 = v29;
          v28 = (v97 - 1) & (v16 + v28);
          v29 = &v95[8 * (unsigned __int64)v28];
          v30 = v29[3];
          if ( v92 == v30 )
            goto LABEL_34;
          v16 = (unsigned int)(v16 + 1);
        }
        if ( !v22 )
          v22 = v29;
        ++v94;
        v23 = v96 + 1;
        if ( 4 * ((int)v96 + 1) < 3 * v97 )
        {
          if ( v97 - HIDWORD(v96) - v23 > v97 >> 3 )
            goto LABEL_23;
          sub_CF32C0((__int64)&v94, v97);
          if ( v97 )
          {
            v60 = v97 - 1;
            v61 = 1;
            v20 = 0;
            v21 = v92;
            v62 = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
            v22 = &v95[8 * (unsigned __int64)v62];
            v63 = v22[3];
            if ( v63 != v92 )
            {
              while ( v63 != -4096 )
              {
                if ( v20 || v63 != -8192 )
                  v22 = (_QWORD *)v20;
                v16 = (unsigned int)(v61 + 1);
                v75 = v62 + v61;
                v62 = v60 & v75;
                v76 = &v95[8 * (unsigned __int64)(v60 & v75)];
                v63 = v76[3];
                if ( v92 == v63 )
                {
                  v22 = &v95[8 * (unsigned __int64)(v60 & v75)];
                  goto LABEL_22;
                }
                v20 = (__int64)v22;
                v61 = v16;
                v22 = v76;
              }
              if ( v20 )
                v22 = (_QWORD *)v20;
            }
LABEL_22:
            v23 = v96 + 1;
LABEL_23:
            v24 = v22[3];
            LODWORD(v96) = v23;
            if ( v24 == -4096 )
            {
              v25 = v22 + 1;
              if ( v21 != -4096 )
                goto LABEL_28;
            }
            else
            {
              --HIDWORD(v96);
              if ( v24 != v21 )
              {
                v25 = v22 + 1;
                if ( v24 != -8192 && v24 )
                {
                  sub_BD60C0(v25);
                  v21 = v92;
                  v25 = v22 + 1;
                }
LABEL_28:
                v22[3] = v21;
                if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
                  sub_BD6050(v25, v90 & 0xFFFFFFFFFFFFFFF8LL);
                v21 = v92;
              }
            }
            v26 = i;
            v27 = v22 + 5;
            *v27 = 6;
            v27[1] = 0;
            *(v27 - 1) = v26;
            v27[2] = 0;
            goto LABEL_35;
          }
LABEL_21:
          v21 = v92;
          v22 = 0;
          goto LABEL_22;
        }
LABEL_20:
        sub_CF32C0((__int64)&v94, 2 * v97);
        if ( v97 )
        {
          v71 = 1;
          v72 = 0;
          v21 = v92;
          v73 = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
          v22 = &v95[8 * (unsigned __int64)v73];
          v74 = v22[3];
          if ( v74 != v92 )
          {
            while ( v74 != -4096 )
            {
              if ( v74 == -8192 && !v72 )
                v72 = v22;
              v16 = (unsigned int)(v71 + 1);
              v73 = (v97 - 1) & (v71 + v73);
              v22 = &v95[8 * (unsigned __int64)v73];
              v74 = v22[3];
              if ( v92 == v74 )
                goto LABEL_22;
              ++v71;
            }
            if ( v72 )
              v22 = v72;
          }
          goto LABEL_22;
        }
        goto LABEL_21;
      }
LABEL_34:
      v27 = v29 + 5;
LABEL_35:
      v89 = &unk_49DB368;
      if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
        sub_BD60C0(&v90);
      v31 = *(unsigned int *)(v19 + 8);
      v32 = v27[2];
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v19 + 12) )
      {
        sub_C8D5F0(v19, (const void *)(v19 + 16), v31 + 1, 8u, v16, v20);
        v31 = *(unsigned int *)(v19 + 8);
      }
      ++v17;
      *(_QWORD *)(*(_QWORD *)v19 + 8 * v31) = v32;
      ++*(_DWORD *)(v19 + 8);
      if ( (__int64 *)v81 == v17 )
        goto LABEL_41;
    }
    ++v94;
    goto LABEL_20;
  }
LABEL_41:
  v33 = *(__int64 **)(a3 + 64);
  v34 = &v33[*(unsigned int *)(a3 + 72)];
  if ( v34 != v33 )
  {
    v35 = *(__int64 **)(a3 + 64);
    while ( 1 )
    {
      v36 = *v35;
      v90 = 2;
      v91 = 0;
      v92 = v36;
      if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
        sub_BD73F0((__int64)&v90);
      i = &v94;
      v89 = &unk_49DD7B0;
      if ( !v97 )
        break;
      v37 = v92;
      v44 = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
      v45 = &v95[8 * (unsigned __int64)v44];
      v46 = v45[3];
      if ( v46 != v92 )
      {
        v16 = 1;
        v38 = 0;
        while ( v46 != -4096 )
        {
          if ( !v38 && v46 == -8192 )
            v38 = v45;
          v44 = (v97 - 1) & (v16 + v44);
          v45 = &v95[8 * (unsigned __int64)v44];
          v46 = v45[3];
          if ( v92 == v46 )
            goto LABEL_61;
          v16 = (unsigned int)(v16 + 1);
        }
        if ( !v38 )
          v38 = v45;
        ++v94;
        v39 = v96 + 1;
        if ( 4 * ((int)v96 + 1) < 3 * v97 )
        {
          if ( v97 - HIDWORD(v96) - v39 > v97 >> 3 )
            goto LABEL_51;
          sub_CF32C0((__int64)&v94, v97);
          if ( v97 )
          {
            v16 = 1;
            v57 = 0;
            v37 = v92;
            v58 = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
            v38 = &v95[8 * (unsigned __int64)v58];
            v59 = v38[3];
            if ( v59 != v92 )
            {
              while ( v59 != -4096 )
              {
                if ( v59 == -8192 && !v57 )
                  v57 = v38;
                v58 = (v97 - 1) & (v16 + v58);
                v38 = &v95[8 * (unsigned __int64)v58];
                v59 = v38[3];
                if ( v92 == v59 )
                  goto LABEL_50;
                v16 = (unsigned int)(v16 + 1);
              }
LABEL_132:
              if ( v57 )
                v38 = v57;
            }
LABEL_50:
            v39 = v96 + 1;
LABEL_51:
            v40 = v38[3];
            LODWORD(v96) = v39;
            if ( v40 == -4096 )
            {
              v41 = v38 + 1;
              if ( v37 != -4096 )
                goto LABEL_56;
            }
            else
            {
              --HIDWORD(v96);
              if ( v40 != v37 )
              {
                v41 = v38 + 1;
                if ( v40 && v40 != -8192 )
                {
                  v79 = v38;
                  v82 = v38 + 1;
                  sub_BD60C0(v41);
                  v37 = v92;
                  v38 = v79;
                  v41 = v82;
                }
LABEL_56:
                v38[3] = v37;
                if ( v37 == 0 || v37 == -4096 || v37 == -8192 )
                {
                  v37 = v92;
                }
                else
                {
                  v83 = v38;
                  sub_BD6050(v41, v90 & 0xFFFFFFFFFFFFFFF8LL);
                  v37 = v92;
                  v38 = v83;
                }
              }
            }
            v42 = i;
            v38[5] = 6;
            v38[6] = 0;
            v38[4] = v42;
            v43 = v38 + 5;
            v38[7] = 0;
            goto LABEL_62;
          }
LABEL_49:
          v37 = v92;
          v38 = 0;
          goto LABEL_50;
        }
LABEL_48:
        sub_CF32C0((__int64)&v94, 2 * v97);
        if ( v97 )
        {
          v68 = 1;
          v57 = 0;
          v37 = v92;
          v69 = (v97 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
          v38 = &v95[8 * (unsigned __int64)v69];
          v70 = v38[3];
          if ( v70 != v92 )
          {
            while ( v70 != -4096 )
            {
              if ( v57 || v70 != -8192 )
                v38 = v57;
              v69 = (v97 - 1) & (v68 + v69);
              v16 = (__int64)&v95[8 * (unsigned __int64)v69];
              v70 = *(_QWORD *)(v16 + 24);
              if ( v92 == v70 )
              {
                v38 = &v95[8 * (unsigned __int64)v69];
                goto LABEL_50;
              }
              ++v68;
              v57 = v38;
              v38 = &v95[8 * (unsigned __int64)v69];
            }
            goto LABEL_132;
          }
          goto LABEL_50;
        }
        goto LABEL_49;
      }
LABEL_61:
      v43 = v45 + 5;
LABEL_62:
      v89 = &unk_49DB368;
      if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
      {
        v84 = v43;
        sub_BD60C0(&v90);
        v43 = v84;
      }
      v47 = *(_QWORD *)(a1 + 120);
      v48 = v43[2];
      v49 = *(unsigned int *)(v47 + 72);
      if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(v47 + 76) )
      {
        v80 = v48;
        v85 = *(_QWORD *)(a1 + 120);
        sub_C8D5F0(v47 + 64, (const void *)(v47 + 80), v49 + 1, 8u, v16, v49 + 1);
        v47 = v85;
        v48 = v80;
        v49 = *(unsigned int *)(v85 + 72);
      }
      ++v35;
      *(_QWORD *)(*(_QWORD *)(v47 + 64) + 8 * v49) = v48;
      ++*(_DWORD *)(v47 + 72);
      if ( v34 == v35 )
        goto LABEL_68;
    }
    ++v94;
    goto LABEL_48;
  }
LABEL_68:
  sub_BD84D0(a2, *(_QWORD *)(a1 + 8));
  if ( v100 )
  {
    v64 = v99;
    v100 = 0;
    if ( v99 )
    {
      v65 = v98;
      v66 = &v98[2 * v99];
      do
      {
        if ( *v65 != -8192 && *v65 != -4096 )
        {
          v67 = v65[1];
          if ( v67 )
            sub_B91220((__int64)(v65 + 1), v67);
        }
        v65 += 2;
      }
      while ( v66 != v65 );
      v64 = v99;
    }
    sub_C7D6A0((__int64)v98, 16LL * v64, 8);
  }
  v50 = v97;
  if ( v97 )
  {
    v52 = v95;
    v86[0] = 2;
    v86[1] = 0;
    v53 = &v95[8 * (unsigned __int64)v97];
    v87 = -4096;
    v89 = &unk_49DD7B0;
    v54 = -4096;
    v88 = 0;
    v90 = 2;
    v91 = 0;
    v92 = -8192;
    i = 0;
    while ( 1 )
    {
      v55 = v52[3];
      if ( v55 != v54 )
      {
        v54 = v92;
        if ( v55 != v92 )
        {
          v56 = v52[7];
          if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
          {
            sub_BD60C0(v52 + 5);
            v55 = v52[3];
          }
          v54 = v55;
        }
      }
      *v52 = &unk_49DB368;
      if ( v54 != 0 && v54 != -4096 && v54 != -8192 )
        sub_BD60C0(v52 + 1);
      v52 += 8;
      if ( v53 == v52 )
        break;
      v54 = v87;
    }
    v89 = &unk_49DB368;
    if ( v92 != -4096 && v92 != 0 && v92 != -8192 )
      sub_BD60C0(&v90);
    if ( v87 != 0 && v87 != -4096 && v87 != -8192 )
      sub_BD60C0(v86);
    v50 = v97;
  }
  return sub_C7D6A0((__int64)v95, v50 << 6, 8);
}
