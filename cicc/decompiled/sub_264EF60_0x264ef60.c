// Function: sub_264EF60
// Address: 0x264ef60
//
void __fastcall sub_264EF60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdx
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // r13
  int v16; // edx
  __int64 v17; // rsi
  __int64 v18; // rcx
  int v19; // edi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 *v24; // r12
  __int64 v25; // rdx
  __int64 **v26; // r13
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 *v29; // r12
  int v30; // r8d
  __int64 v31; // rcx
  int v32; // r8d
  int v33; // r9d
  unsigned int i; // eax
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 *v39; // rsi
  __int64 v40; // rbx
  unsigned int v41; // esi
  int v42; // edx
  _QWORD *v43; // rax
  int v44; // edx
  __int64 v45; // rdx
  _QWORD *v46; // rax
  __int64 *v47; // rax
  const __m128i *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdi
  int v51; // r10d
  unsigned int j; // eax
  __int64 v53; // r8
  unsigned int v54; // eax
  unsigned int v55; // esi
  int v56; // edx
  __int64 v57; // rax
  int v58; // edx
  __int32 v59; // edx
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // rbx
  __int64 v63; // r11
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 *v69; // rcx
  __int64 *v70; // r14
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 *v73; // r13
  __int64 k; // rbx
  __int64 v75; // rdx
  char v76; // al
  __int64 v77; // r12
  _QWORD *v78; // rax
  _QWORD *v79; // rsi
  _QWORD *v80; // r8
  __int64 *v81; // rax
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // r14
  __int64 v85; // rax
  __m128i v86; // xmm0
  __int64 *v87; // r14
  __int64 *m; // r12
  __int64 v89; // rdi
  int v90; // r8d
  __int64 v91; // [rsp+0h] [rbp-150h]
  __int64 **v92; // [rsp+8h] [rbp-148h]
  __int64 v93; // [rsp+10h] [rbp-140h]
  __int64 v94; // [rsp+18h] [rbp-138h]
  __int64 v95; // [rsp+28h] [rbp-128h]
  __int64 v96; // [rsp+38h] [rbp-118h]
  __int64 v97; // [rsp+38h] [rbp-118h]
  __int64 *v98; // [rsp+38h] [rbp-118h]
  __m128i v99; // [rsp+40h] [rbp-110h] BYREF
  __int64 v100; // [rsp+50h] [rbp-100h]
  __int64 v101[2]; // [rsp+58h] [rbp-F8h] BYREF
  const __m128i *v102; // [rsp+68h] [rbp-E8h] BYREF
  __m128i v103; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v104; // [rsp+80h] [rbp-D0h] BYREF
  unsigned __int64 v105; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v106; // [rsp+98h] [rbp-B8h]
  _QWORD v107[4]; // [rsp+B0h] [rbp-A0h] BYREF
  __m128i v108; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v109; // [rsp+E0h] [rbp-70h]
  __int64 v110; // [rsp+E8h] [rbp-68h]
  _BYTE v111[96]; // [rsp+F0h] [rbp-60h] BYREF

  v101[0] = a2;
  v99.m128i_i64[0] = (__int64)v101;
  sub_26463C0((__int64)v111, a3, v101);
  if ( !v111[32] )
    return;
  sub_2640E50(&v105, (_QWORD *)(v101[0] + 72), v9);
  v10 = v105;
  v100 = v106;
  if ( v105 != v106 )
  {
    v11 = a5;
    v12 = a3;
    v13 = v11;
    do
    {
      v14 = *(_QWORD *)(*(_QWORD *)v10 + 8LL);
      if ( **(_QWORD **)v10 || v14 )
        sub_264EF60(a1, v14, v12, a4, v13);
      v10 += 16LL;
    }
    while ( v100 != v10 );
    a5 = v13;
  }
  v15 = v101[0];
  if ( *(_BYTE *)v101[0] || (v16 = *(_DWORD *)(a4 + 24), v17 = *(_QWORD *)(a4 + 8), !v16) )
  {
LABEL_26:
    sub_2644030(&v105);
    return;
  }
  v18 = *(_QWORD *)(v101[0] + 40);
  v19 = v16 - 1;
  v20 = (v16 - 1) & ((unsigned int)((0xBF58476D1CE4E5B9LL * v18) >> 31) ^ (484763065 * (_DWORD)v18));
  v21 = *(_QWORD *)(v17 + 32 * v20);
  if ( v18 != v21 )
  {
    v90 = 1;
    while ( v21 != -1 )
    {
      LODWORD(v20) = v19 & (v90 + v20);
      v21 = *(_QWORD *)(v17 + 32LL * (unsigned int)v20);
      if ( v18 == v21 )
        goto LABEL_11;
      ++v90;
    }
    goto LABEL_26;
  }
LABEL_11:
  if ( (unsigned __int8)sub_263E020(a4, (__int64 *)(v101[0] + 40), v107) )
  {
    v22 = *(_QWORD *)(v107[0] + 16LL);
    v23 = v107[0] + 8LL;
    goto LABEL_13;
  }
  v41 = *(_DWORD *)(a4 + 24);
  v42 = *(_DWORD *)(a4 + 16);
  v43 = (_QWORD *)v107[0];
  ++*(_QWORD *)a4;
  v44 = v42 + 1;
  v108.m128i_i64[0] = (__int64)v43;
  if ( 4 * v44 >= 3 * v41 )
  {
    v41 *= 2;
    goto LABEL_97;
  }
  if ( v41 - *(_DWORD *)(a4 + 20) - v44 <= v41 >> 3 )
  {
LABEL_97:
    sub_2645370(a4, v41);
    sub_263E020(a4, (__int64 *)(v15 + 40), &v108);
    v44 = *(_DWORD *)(a4 + 16) + 1;
    v43 = (_QWORD *)v108.m128i_i64[0];
  }
  *(_DWORD *)(a4 + 16) = v44;
  if ( *v43 != -1 )
    --*(_DWORD *)(a4 + 20);
  v45 = *(_QWORD *)(v15 + 40);
  v46 = v43 + 1;
  *v46 = 0;
  v46[1] = 0;
  *(v46 - 1) = v45;
  v46[2] = 0;
  v23 = (__int64)v46;
  v22 = 0;
LABEL_13:
  v24 = *(__int64 **)v23;
  v25 = v101[0];
  v94 = v101[0];
  if ( v22 - *(_QWORD *)v23 == 72 && v24[2] - v24[1] == 8 )
  {
    if ( !*(_BYTE *)(v101[0] + 1) )
    {
      v37 = *v24;
      *(_DWORD *)(v101[0] + 16) = 0;
      *(_QWORD *)(v25 + 8) = v37;
      v38 = *v24;
      v108.m128i_i32[2] = 0;
      v108.m128i_i64[0] = v38;
      v39 = (unsigned __int64 *)v99.m128i_i64[0];
      *(_QWORD *)sub_2645C20(a1 + 272, &v108) = v25;
      v40 = v24[4];
      *(_QWORD *)sub_263EE40((_QWORD *)(a1 + 48), v39) = v40;
    }
    goto LABEL_26;
  }
  v26 = (__int64 **)v23;
  sub_264D230((__int64)v107, v101[0], v101[0]);
  v27 = *(__int64 **)v23;
  v28 = 0;
  LODWORD(v100) = 0;
  if ( *(__int64 **)(v23 + 8) != v27 )
  {
    while ( 1 )
    {
      v29 = &v27[9 * v28];
      if ( *((_DWORD *)v29 + 14) )
        break;
      v30 = *(_DWORD *)(a5 + 24);
      v31 = *v29;
      if ( v30 )
      {
        v32 = v30 - 1;
        v33 = 1;
        for ( i = v32 & (969526130 * (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4))); ; i = v32 & v36 )
        {
          v35 = *(_QWORD *)(a5 + 8) + 32LL * i;
          if ( v31 == *(_QWORD *)v35 && !*(_DWORD *)(v35 + 8) )
            break;
          if ( *(_QWORD *)v35 == -4096 && *(_DWORD *)(v35 + 8) == -1 )
            goto LABEL_33;
          v36 = v33 + i;
          ++v33;
        }
        v108.m128i_i64[0] = *v29;
        v108.m128i_i32[2] = 0;
        if ( (unsigned __int8)sub_26403F0(a5, v108.m128i_i64, &v102) )
        {
          v48 = v102 + 1;
          goto LABEL_38;
        }
        v55 = *(_DWORD *)(a5 + 24);
        v56 = *(_DWORD *)(a5 + 16);
        v57 = (__int64)v102;
        ++*(_QWORD *)a5;
        v58 = v56 + 1;
        v103.m128i_i64[0] = v57;
        if ( 4 * v58 >= 3 * v55 )
        {
          v55 *= 2;
        }
        else if ( v55 - *(_DWORD *)(a5 + 20) - v58 > v55 >> 3 )
        {
          goto LABEL_48;
        }
        sub_2645E10(a5, v55);
        sub_26403F0(a5, v108.m128i_i64, &v103);
        v58 = *(_DWORD *)(a5 + 16) + 1;
        v57 = v103.m128i_i64[0];
LABEL_48:
        *(_DWORD *)(a5 + 16) = v58;
        if ( *(_QWORD *)v57 != -4096 || *(_DWORD *)(v57 + 8) != -1 )
          --*(_DWORD *)(a5 + 20);
        v48 = (const __m128i *)(v57 + 16);
        v48[-1].m128i_i64[0] = v108.m128i_i64[0];
        v59 = v108.m128i_i32[2];
        v48->m128i_i64[0] = 0;
        v48[-1].m128i_i32[2] = v59;
        v48->m128i_i32[2] = 0;
LABEL_38:
        v49 = *(unsigned int *)(a1 + 296);
        v50 = *(_QWORD *)(a1 + 280);
        v103 = _mm_loadu_si128(v48);
        if ( (_DWORD)v49 )
        {
          v51 = 1;
          for ( j = (v49 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v103.m128i_i32[2])
                      | ((unsigned __int64)(((unsigned __int32)v103.m128i_i32[0] >> 9)
                                          ^ ((unsigned __int32)v103.m128i_i32[0] >> 4)) << 32))) >> 31)
                   ^ (756364221 * v103.m128i_i32[2])); ; j = (v49 - 1) & v54 )
          {
            v53 = v50 + 24LL * j;
            if ( *(_QWORD *)v53 == v103.m128i_i64[0] && *(_DWORD *)(v53 + 8) == v103.m128i_i32[2] )
              break;
            if ( *(_QWORD *)v53 == -4096 && *(_DWORD *)(v53 + 8) == -1 )
              goto LABEL_45;
            v54 = v51 + j;
            ++v51;
          }
          if ( v53 != v50 + 24 * v49 )
          {
            v81 = (__int64 *)sub_2645C20(a1 + 272, &v103);
            v104.m128i_i32[2] = 0;
            v84 = *v81;
            v104.m128i_i64[0] = *v29;
            v85 = *(unsigned int *)(v84 + 32);
            v86 = _mm_load_si128(&v104);
            if ( v85 + 1 > (unsigned __int64)*(unsigned int *)(v84 + 36) )
            {
              v99 = v86;
              sub_C8D5F0(v84 + 24, (const void *)(v84 + 40), v85 + 1, 0x10u, v82, v83);
              v85 = *(unsigned int *)(v84 + 32);
              v86 = _mm_load_si128(&v99);
            }
            *(__m128i *)(*(_QWORD *)(v84 + 24) + 16 * v85) = v86;
            ++*(_DWORD *)(v84 + 32);
            v27 = *v26;
            goto LABEL_33;
          }
        }
LABEL_45:
        v27 = *v26;
      }
LABEL_33:
      v47 = v26[1];
      LODWORD(v100) = v100 + 1;
      v28 = (unsigned int)v100;
      if ( (unsigned int)v100 >= 0x8E38E38E38E38E39LL * (v47 - v27) )
        goto LABEL_34;
    }
    v95 = (__int64)(v29 + 5);
    sub_22B06E0((__int64)(v29 + 5), (__int64)v107);
    v60 = v29[2];
    if ( v60 - 8 != v29[1] )
    {
      v99.m128i_i64[0] = (__int64)v26;
      v61 = v60 - 8;
      v96 = a5;
      do
      {
        v62 = sub_263DED0(a1, *(_QWORD *)(v61 - 8));
        v64 = *(_QWORD *)(v62 + 72);
        v65 = *(_QWORD *)(v62 + 80);
        if ( v64 == v65 )
          goto LABEL_32;
        while ( *(_QWORD *)(*(_QWORD *)v64 + 8LL) != v63 )
        {
          v64 += 16;
          if ( v65 == v64 )
            goto LABEL_32;
        }
        sub_22B06E0(v95, *(_QWORD *)v64 + 24LL);
        if ( !*((_DWORD *)v29 + 14) )
        {
LABEL_32:
          v26 = (__int64 **)v99.m128i_i64[0];
          a5 = v96;
          v27 = *(__int64 **)v99.m128i_i64[0];
          goto LABEL_33;
        }
        v61 -= 8;
      }
      while ( v29[1] != v61 );
      v26 = (__int64 **)v99.m128i_i64[0];
      a5 = v96;
    }
    v66 = *v29;
    v108.m128i_i32[2] = 0;
    v108.m128i_i64[0] = v66;
    v93 = sub_2648220((_QWORD *)a1, 0, v29[4], v66, 0);
    v67 = *v29;
    v99.m128i_i64[0] = (__int64)&v108;
    v108.m128i_i64[0] = v67;
    v108.m128i_i32[2] = 0;
    *(_QWORD *)sub_2645C20(a1 + 272, &v108) = v93;
    *(_BYTE *)(v93 + 2) = sub_26484B0(a1, v95);
    v68 = sub_263DED0(a1, *(_QWORD *)v29[1]);
    v108 = 0u;
    v97 = v68;
    v109 = 0;
    v110 = 0;
    sub_264A680(v99.m128i_i64[0], v95);
    sub_264E850(a1, v93, v97, 1, v99.m128i_i64[0]);
    sub_2342640(v99.m128i_i64[0]);
    v108 = 0u;
    v109 = 0;
    v110 = 0;
    sub_264A680(v99.m128i_i64[0], v95);
    sub_264E850(a1, v93, v94, 0, v99.m128i_i64[0]);
    sub_2342640(v99.m128i_i64[0]);
    v69 = (__int64 *)v29[1];
    v70 = (__int64 *)v29[2];
    if ( v70 != v69 )
    {
      v71 = *v69;
      v99.m128i_i64[0] = v29[1];
      v72 = sub_263DED0(a1, v71);
      v98 = v29;
      v92 = v26;
      v91 = a5;
      v73 = (__int64 *)v99.m128i_i64[0];
      for ( k = v72; ; k = v77 )
      {
        v75 = *(_QWORD *)(k + 48);
        v76 = 0;
        if ( *(_QWORD *)(k + 56) != v75 )
          v76 = sub_2647F70(k, v71, v75);
        ++v73;
        *(_BYTE *)(k + 2) = v76;
        if ( v70 == v73 )
          break;
        v77 = sub_263DED0(a1, *v73);
        v78 = *(_QWORD **)(v77 + 48);
        v79 = *(_QWORD **)(v77 + 56);
        if ( v78 == v79 )
        {
LABEL_98:
          sub_2649AE0(24, v95);
          BUG();
        }
        while ( 1 )
        {
          v80 = (_QWORD *)*v78;
          if ( k == *(_QWORD *)*v78 )
            break;
          v78 += 2;
          if ( v79 == v78 )
            goto LABEL_98;
        }
        v71 = v95;
        v99.m128i_i64[0] = *v78;
        sub_2649AE0((__int64)(v80 + 3), v95);
        if ( !*(_DWORD *)(v99.m128i_i64[0] + 40) )
        {
          v71 = 0;
          sub_264E780(v99.m128i_i64[0], 0, 1);
        }
      }
      v29 = v98;
      v26 = v92;
      a5 = v91;
    }
    if ( (_BYTE)qword_4FF39C8 )
    {
      if ( *(_BYTE *)(v93 + 2) )
        sub_264C780((_QWORD *)v93);
      v87 = (__int64 *)v29[1];
      for ( m = (__int64 *)v29[2]; m != v87; ++v87 )
      {
        v89 = sub_263DED0(a1, *v87);
        if ( *(_BYTE *)(v89 + 2) )
          sub_264C780((_QWORD *)v89);
      }
    }
    goto LABEL_45;
  }
LABEL_34:
  sub_2342640((__int64)v107);
  sub_2644030(&v105);
}
