// Function: sub_3571960
// Address: 0x3571960
//
__int64 __fastcall sub_3571960(__int64 *a1, __int64 a2, char a3)
{
  unsigned __int64 v3; // rbx
  char v5; // al
  char v6; // r11
  unsigned int v7; // r15d
  char v8; // r12
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // r11
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 *v19; // r14
  __int64 v21; // r10
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  _BYTE *v24; // r12
  unsigned int v25; // esi
  __int64 v26; // r8
  int v27; // r11d
  __int64 v28; // rcx
  unsigned int v29; // edx
  _QWORD *v30; // rax
  __int64 v31; // r9
  _DWORD *v32; // rax
  int v33; // edx
  unsigned int v34; // edx
  int v35; // eax
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // r11
  __int64 v40; // r14
  _QWORD *v41; // rbx
  __int64 v42; // rdx
  __int64 v43; // r13
  _BYTE *v44; // r15
  unsigned __int64 v45; // r14
  __int32 v46; // edx
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rbx
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r11
  int v57; // r10d
  unsigned int v58; // edx
  __int64 v59; // r8
  int v60; // r10d
  unsigned int i; // ebx
  int v62; // r9d
  __int64 v63; // rdx
  unsigned int v64; // edi
  __int64 v65; // rax
  __int64 v66; // rcx
  int v67; // eax
  __int64 v68; // rax
  int v69; // r15d
  __int64 v70; // r12
  unsigned int v71; // ecx
  int v72; // eax
  __int64 v73; // r8
  __int64 v74; // r8
  unsigned int v75; // r13d
  int v76; // r9d
  __int64 v77; // rsi
  __int64 v78; // rax
  unsigned int v79; // ebx
  int v80; // r11d
  __int64 v81; // r9
  int v82; // r10d
  unsigned __int64 v83; // [rsp+8h] [rbp-128h]
  int v84; // [rsp+18h] [rbp-118h]
  int v85; // [rsp+18h] [rbp-118h]
  int v86; // [rsp+18h] [rbp-118h]
  char v89; // [rsp+28h] [rbp-108h]
  __int64 *v91; // [rsp+30h] [rbp-100h]
  __int64 v92; // [rsp+30h] [rbp-100h]
  __int64 *v93; // [rsp+38h] [rbp-F8h]
  __int64 *v94; // [rsp+38h] [rbp-F8h]
  __m128i v95; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v96; // [rsp+50h] [rbp-E0h]
  __int64 v97; // [rsp+58h] [rbp-D8h]
  __int64 v98; // [rsp+60h] [rbp-D0h]
  _BYTE *v99; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v100; // [rsp+78h] [rbp-B8h]
  _BYTE v101[176]; // [rsp+80h] [rbp-B0h] BYREF

  v5 = sub_356E570(a1, a2);
  v6 = a3;
  if ( !v5 )
    return sub_3571720(a1, a2, a3);
  v7 = *(_DWORD *)(a2 + 72);
  if ( !v7 )
  {
    if ( !a3 )
    {
      v52 = a1[4];
      v53 = a1[5];
      v54 = (__int64 *)sub_2E313E0(a2);
      sub_356E080(0xAu, a2, v54, a1[1], a1[2], v53, v52);
      return *(unsigned int *)(*(_QWORD *)(v55 + 32) + 8LL);
    }
    return v7;
  }
  v89 = v6;
  v8 = v5;
  v99 = v101;
  v100 = 0x800000000LL;
  v9 = *(__int64 **)(a2 + 64);
  v10 = v7;
  v7 = 0;
  v91 = &v9[v10];
  do
  {
    while ( 1 )
    {
      v11 = *v9;
      v12 = (unsigned int)sub_3571720(a1, *v9, v89);
      v14 = v12 | v3 & 0xFFFFFFFF00000000LL;
      v15 = (unsigned int)v100;
      v3 = v14;
      v16 = (unsigned int)v100 + 1LL;
      if ( v16 > HIDWORD(v100) )
      {
        v83 = v14;
        v84 = v12;
        sub_C8D5F0((__int64)&v99, v101, v16, 0x10u, v12, v13);
        v15 = (unsigned int)v100;
        v14 = v83;
        LODWORD(v12) = v84;
      }
      v17 = (__int64 *)&v99[16 * v15];
      *v17 = v11;
      v17[1] = v14;
      v18 = (unsigned int)(v100 + 1);
      LODWORD(v100) = v100 + 1;
      if ( !v8 )
        break;
      v7 = v12;
      ++v9;
      v8 = 0;
      if ( v91 == v9 )
        goto LABEL_11;
    }
    if ( (_DWORD)v12 != v7 )
      v7 = 0;
    ++v9;
    v8 = 0;
  }
  while ( v91 != v9 );
LABEL_11:
  v19 = a1;
  if ( v7 )
    goto LABEL_12;
  v93 = (__int64 *)(a2 + 48);
  if ( a2 + 48 == (*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_41;
  v21 = *(_QWORD *)(a2 + 56);
  if ( *(_WORD *)(v21 + 68) != 68 )
  {
    if ( *(_WORD *)(v21 + 68) )
      goto LABEL_41;
  }
  v22 = (unsigned __int64)v99;
  v23 = 16 * v18;
  v95 = 0u;
  v96 = 0;
  v24 = &v99[v23];
  LODWORD(v97) = 0;
  if ( v99 == &v99[v23] )
  {
    v38 = 0;
    v39 = 0;
    goto LABEL_35;
  }
  v92 = v21;
  v25 = 0;
  v26 = 0;
  while ( 1 )
  {
    if ( v25 )
    {
      v27 = 1;
      v28 = 0;
      v29 = (v25 - 1) & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
      v30 = (_QWORD *)(v26 + 16LL * v29);
      v31 = *v30;
      if ( *(_QWORD *)v22 == *v30 )
      {
LABEL_24:
        v32 = v30 + 1;
        goto LABEL_25;
      }
      while ( v31 != -4096 )
      {
        if ( !v28 && v31 == -8192 )
          v28 = (__int64)v30;
        v29 = (v25 - 1) & (v27 + v29);
        v30 = (_QWORD *)(v26 + 16LL * v29);
        v31 = *v30;
        if ( *(_QWORD *)v22 == *v30 )
          goto LABEL_24;
        ++v27;
      }
      if ( !v28 )
        v28 = (__int64)v30;
      ++v95.m128i_i64[0];
      v35 = v96 + 1;
      if ( 4 * ((int)v96 + 1) < 3 * v25 )
      {
        if ( v25 - (v35 + HIDWORD(v96)) > v25 >> 3 )
          goto LABEL_31;
        sub_34F9190((__int64)&v95, v25);
        if ( !(_DWORD)v97 )
        {
LABEL_136:
          LODWORD(v96) = v96 + 1;
          BUG();
        }
        v56 = 0;
        v57 = 1;
        v58 = (v97 - 1) & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
        v35 = v96 + 1;
        v28 = v95.m128i_i64[1] + 16LL * v58;
        v59 = *(_QWORD *)v28;
        if ( *(_QWORD *)v28 == *(_QWORD *)v22 )
          goto LABEL_31;
        while ( v59 != -4096 )
        {
          if ( v59 == -8192 && !v56 )
            v56 = v28;
          v58 = (v97 - 1) & (v57 + v58);
          v28 = v95.m128i_i64[1] + 16LL * v58;
          v59 = *(_QWORD *)v28;
          if ( *(_QWORD *)v22 == *(_QWORD *)v28 )
            goto LABEL_31;
          ++v57;
        }
        goto LABEL_67;
      }
    }
    else
    {
      ++v95.m128i_i64[0];
    }
    sub_34F9190((__int64)&v95, 2 * v25);
    if ( !(_DWORD)v97 )
      goto LABEL_136;
    v34 = (v97 - 1) & (((unsigned int)*(_QWORD *)v22 >> 9) ^ ((unsigned int)*(_QWORD *)v22 >> 4));
    v35 = v96 + 1;
    v28 = v95.m128i_i64[1] + 16LL * v34;
    v36 = *(_QWORD *)v28;
    if ( *(_QWORD *)v22 == *(_QWORD *)v28 )
      goto LABEL_31;
    v82 = 1;
    v56 = 0;
    while ( v36 != -4096 )
    {
      if ( !v56 && v36 == -8192 )
        v56 = v28;
      v34 = (v97 - 1) & (v82 + v34);
      v28 = v95.m128i_i64[1] + 16LL * v34;
      v36 = *(_QWORD *)v28;
      if ( *(_QWORD *)v22 == *(_QWORD *)v28 )
        goto LABEL_31;
      ++v82;
    }
LABEL_67:
    if ( v56 )
      v28 = v56;
LABEL_31:
    LODWORD(v96) = v35;
    if ( *(_QWORD *)v28 != -4096 )
      --HIDWORD(v96);
    v37 = *(_QWORD *)v22;
    *(_DWORD *)(v28 + 8) = 0;
    *(_QWORD *)v28 = v37;
    v32 = (_DWORD *)(v28 + 8);
LABEL_25:
    v33 = *(_DWORD *)(v22 + 8);
    v22 += 16LL;
    *v32 = v33;
    if ( v24 == (_BYTE *)v22 )
      break;
    v26 = v95.m128i_i64[1];
    v25 = v97;
  }
  v21 = v92;
  v38 = (unsigned int)v97;
  v39 = v95.m128i_i64[1];
LABEL_35:
  if ( v93 == (__int64 *)v21 )
    goto LABEL_40;
  v40 = v21;
LABEL_37:
  if ( *(_WORD *)(v40 + 68) != 68 && *(_WORD *)(v40 + 68) )
    goto LABEL_39;
  v60 = *(_DWORD *)(v40 + 40) & 0xFFFFFF;
  if ( v60 != 1 )
  {
    for ( i = 1; i != v60; i += 2 )
    {
      v68 = *(_QWORD *)(v40 + 32);
      v69 = *(_DWORD *)(v68 + 40LL * i + 8);
      v70 = *(_QWORD *)(v68 + 40LL * (i + 1) + 24);
      if ( (_DWORD)v38 )
      {
        v62 = 1;
        v63 = 0;
        v64 = (v38 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
        v65 = v39 + 16LL * v64;
        v66 = *(_QWORD *)v65;
        if ( v70 == *(_QWORD *)v65 )
        {
LABEL_73:
          v67 = *(_DWORD *)(v65 + 8);
          goto LABEL_74;
        }
        while ( v66 != -4096 )
        {
          if ( v66 == -8192 && !v63 )
            v63 = v65;
          v64 = (v38 - 1) & (v62 + v64);
          v65 = v39 + 16LL * v64;
          v66 = *(_QWORD *)v65;
          if ( v70 == *(_QWORD *)v65 )
            goto LABEL_73;
          ++v62;
        }
        if ( !v63 )
          v63 = v65;
        ++v95.m128i_i64[0];
        v72 = v96 + 1;
        if ( 4 * ((int)v96 + 1) < (unsigned int)(3 * v38) )
        {
          if ( (int)v38 - (v72 + HIDWORD(v96)) <= (unsigned int)v38 >> 3 )
          {
            v86 = v60;
            sub_34F9190((__int64)&v95, v38);
            if ( !(_DWORD)v97 )
              goto LABEL_136;
            v74 = 0;
            v75 = (v97 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
            v60 = v86;
            v76 = 1;
            v72 = v96 + 1;
            v63 = v95.m128i_i64[1] + 16LL * v75;
            v77 = *(_QWORD *)v63;
            if ( v70 != *(_QWORD *)v63 )
            {
              while ( v77 != -4096 )
              {
                if ( !v74 && v77 == -8192 )
                  v74 = v63;
                v75 = (v97 - 1) & (v76 + v75);
                v63 = v95.m128i_i64[1] + 16LL * v75;
                v77 = *(_QWORD *)v63;
                if ( v70 == *(_QWORD *)v63 )
                  goto LABEL_80;
                ++v76;
              }
              if ( v74 )
                v63 = v74;
            }
          }
          goto LABEL_80;
        }
      }
      else
      {
        ++v95.m128i_i64[0];
      }
      v85 = v60;
      sub_34F9190((__int64)&v95, 2 * v38);
      if ( !(_DWORD)v97 )
        goto LABEL_136;
      v60 = v85;
      v71 = (v97 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
      v72 = v96 + 1;
      v63 = v95.m128i_i64[1] + 16LL * v71;
      v73 = *(_QWORD *)v63;
      if ( v70 != *(_QWORD *)v63 )
      {
        v80 = 1;
        v81 = 0;
        while ( v73 != -4096 )
        {
          if ( !v81 && v73 == -8192 )
            v81 = v63;
          v71 = (v97 - 1) & (v80 + v71);
          v63 = v95.m128i_i64[1] + 16LL * v71;
          v73 = *(_QWORD *)v63;
          if ( v70 == *(_QWORD *)v63 )
            goto LABEL_80;
          ++v80;
        }
        if ( v81 )
          v63 = v81;
      }
LABEL_80:
      LODWORD(v96) = v72;
      if ( *(_QWORD *)v63 != -4096 )
        --HIDWORD(v96);
      *(_QWORD *)v63 = v70;
      v67 = 0;
      *(_DWORD *)(v63 + 8) = 0;
LABEL_74:
      if ( v69 != v67 )
      {
        if ( (*(_BYTE *)v40 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v40 + 44) & 8) != 0 )
            v40 = *(_QWORD *)(v40 + 8);
        }
        v40 = *(_QWORD *)(v40 + 8);
        v38 = (unsigned int)v97;
        v39 = v95.m128i_i64[1];
        if ( v93 == (__int64 *)v40 )
        {
LABEL_39:
          v7 = 0;
          v19 = a1;
LABEL_40:
          sub_C7D6A0(v39, 16 * v38, 8);
LABEL_41:
          if ( !a3 )
          {
            if ( v93 != (__int64 *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
              v93 = *(__int64 **)(a2 + 56);
            v41 = sub_356E080(0, a2, v93, v19[1], v19[2], v19[5], v19[4]);
            v43 = v42;
            if ( v99 != &v99[16 * (unsigned int)v100] )
            {
              v94 = v19;
              v44 = &v99[16 * (unsigned int)v100];
              v45 = (unsigned __int64)v99;
              do
              {
                v46 = *(_DWORD *)(v45 + 8);
                v45 += 16LL;
                v95.m128i_i64[0] = 0;
                v95.m128i_i32[2] = v46;
                v96 = 0;
                v97 = 0;
                v98 = 0;
                sub_2E8EAD0(v43, (__int64)v41, &v95);
                v47 = *(_QWORD *)(v45 - 16);
                v95.m128i_i8[0] = 4;
                v97 = v47;
                v95.m128i_i32[0] &= 0xFFF000FF;
                v96 = 0;
                sub_2E8EAD0(v43, (__int64)v41, &v95);
              }
              while ( v44 != (_BYTE *)v45 );
              v19 = v94;
            }
            v7 = sub_2E8B030(v43);
            if ( v7 )
            {
              sub_2E88E20(v43);
            }
            else
            {
              v50 = v19[3];
              if ( v50 )
              {
                v51 = *(unsigned int *)(v50 + 8);
                if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 12) )
                {
                  sub_C8D5F0(v19[3], (const void *)(v50 + 16), v51 + 1, 8u, v48, v49);
                  v51 = *(unsigned int *)(v50 + 8);
                }
                *(_QWORD *)(*(_QWORD *)v50 + 8 * v51) = v43;
                ++*(_DWORD *)(v50 + 8);
              }
              v7 = *(_DWORD *)(*(_QWORD *)(v43 + 32) + 8LL);
            }
          }
          goto LABEL_12;
        }
        goto LABEL_37;
      }
      v38 = (unsigned int)v97;
      v39 = v95.m128i_i64[1];
    }
  }
  v78 = *(_QWORD *)(v40 + 32);
  v7 = 0;
  v19 = a1;
  v79 = *(_DWORD *)(v78 + 8);
  sub_C7D6A0(v39, 16 * v38, 8);
  if ( !v79 )
    goto LABEL_41;
  v7 = v79;
LABEL_12:
  if ( v99 != v101 )
    _libc_free((unsigned __int64)v99);
  return v7;
}
