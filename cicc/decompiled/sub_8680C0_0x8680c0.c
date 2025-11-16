// Function: sub_8680C0
// Address: 0x8680c0
//
const __m128i *__fastcall sub_8680C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        int a5,
        int a6,
        __int64 a7,
        int *a8)
{
  const __m128i *v8; // r12
  __int64 v9; // rdi
  const __m128i *k; // r13
  _QWORD *m128i_i64; // rbx
  __m128i *v12; // rax
  __int32 v13; // eax
  _QWORD *v14; // r12
  __int64 *v15; // r15
  _QWORD *v16; // r14
  _QWORD *v17; // rbx
  int v18; // r13d
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rax
  int i; // edx
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 v26; // rax
  __int32 v27; // esi
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r15
  const __m128i *v34; // rax
  const __m128i *v35; // rdx
  __int8 v36; // si
  char v37; // bl
  bool v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 *v41; // rax
  __int64 *v42; // rcx
  const char *v43; // r15
  __int64 v44; // r14
  __int64 v45; // r12
  __int64 v46; // rdx
  int v47; // eax
  _QWORD *v48; // rax
  __int64 v49; // rax
  __int64 v50; // rbx
  const char *v51; // rsi
  __int64 v52; // rax
  __int8 v53; // cl
  __int64 v54; // rax
  char j; // dl
  _QWORD *v56; // r15
  _QWORD *v57; // rbx
  __int64 **v58; // r14
  bool v59; // r12
  _QWORD *v60; // rax
  __int64 *v61; // rcx
  __int64 v62; // r15
  __int8 *v63; // rax
  __m128i v64; // xmm0
  __m128i v65; // xmm2
  __m128i v66; // xmm4
  __m128i v67; // xmm6
  __m128i v68; // xmm0
  __m128i *v69; // r15
  const __m128i *v70; // rax
  __int64 v71; // rax
  __int64 v72; // r12
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 *v76; // r9
  int v77; // r11d
  _QWORD *v79; // [rsp+10h] [rbp-110h]
  _QWORD *v80; // [rsp+10h] [rbp-110h]
  _QWORD *v81; // [rsp+10h] [rbp-110h]
  const __m128i *v82; // [rsp+20h] [rbp-100h]
  _QWORD *v83; // [rsp+20h] [rbp-100h]
  char v87; // [rsp+40h] [rbp-E0h]
  int v89; // [rsp+48h] [rbp-D8h]
  int v90; // [rsp+4Ch] [rbp-D4h]
  int v91; // [rsp+50h] [rbp-D0h]
  __m128i *v93; // [rsp+58h] [rbp-C8h]
  int v94; // [rsp+68h] [rbp-B8h] BYREF
  int v95; // [rsp+6Ch] [rbp-B4h] BYREF
  const __m128i *v96; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+78h] [rbp-A8h]
  __int64 v98; // [rsp+80h] [rbp-A0h]
  _QWORD v99[2]; // [rsp+90h] [rbp-90h] BYREF
  __m128i v100; // [rsp+A0h] [rbp-80h]
  __m128i v101; // [rsp+B0h] [rbp-70h]
  __m128i v102; // [rsp+C0h] [rbp-60h]
  __m128i v103; // [rsp+D0h] [rbp-50h]
  __m128i v104; // [rsp+E0h] [rbp-40h]

  v8 = *(const __m128i **)(a1 + 24);
  if ( !v8 )
  {
    v89 = a6 | a5;
    if ( !(a6 | a5) )
      goto LABEL_44;
    v93 = 0;
    v37 = 0;
    v38 = 1;
    v89 = 0;
    goto LABEL_138;
  }
  v94 = 0;
  v9 = (__int64)v8;
  k = v8;
  v90 = 0;
  v93 = sub_8665B0(v8);
  m128i_i64 = v93->m128i_i64;
  v91 = 1;
  v89 = 0;
  v87 = 0;
  while ( 1 )
  {
    v13 = k[2].m128i_i32[0];
    if ( a5 )
    {
      if ( !v13 && !k[6].m128i_i8[0] )
      {
        v33 = k->m128i_i64[1];
        sub_89ED70(a2, a3, v99, &v96);
        v34 = v96;
        if ( !v96 )
          goto LABEL_174;
        while ( v33 != *(_QWORD *)(v99[0] + 8LL) )
        {
          sub_89ED80(v99, &v96);
          v34 = v96;
          if ( !v96 )
LABEL_174:
            BUG();
        }
        v35 = (const __m128i *)v34->m128i_i64[0];
        if ( v34->m128i_i64[0] )
        {
          v36 = v35[1].m128i_i8[8];
          if ( (v36 & 8) != 0 )
          {
            while ( (v36 & 0xA) == 0xA )
            {
              if ( !v35->m128i_i64[0] )
              {
                v34 = v35;
                goto LABEL_76;
              }
              v36 = *(_BYTE *)(v35->m128i_i64[0] + 24);
              v34 = v35;
              v35 = (const __m128i *)v35->m128i_i64[0];
            }
            if ( (v36 & 8) == 0 )
              v35 = 0;
          }
          else
          {
LABEL_76:
            v35 = 0;
          }
        }
        m128i_i64[11] = v34;
        m128i_i64[10] = v35;
      }
      goto LABEL_5;
    }
    if ( v13 == 1 )
    {
      v27 = k[1].m128i_i32[0];
      i = k[3].m128i_i32[0];
      v28 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v9 = *(unsigned __int8 *)(v28 + 4);
      if ( (_BYTE)v9 == 9 && (*(_BYTE *)(v28 + 12) & 0x10) != 0 )
      {
        v9 = *(unsigned __int8 *)(v28 - 772);
        if ( (_BYTE)v9 == 10 && (*(_BYTE *)(v28 - 764) & 0x10) != 0 )
        {
          v9 = *(unsigned __int8 *)(v28 - 1548);
          v28 -= 1552;
        }
        else
        {
          v28 -= 776;
        }
      }
      if ( (_BYTE)v9 != 17 )
      {
LABEL_54:
        v29 = *(int *)(v28 + 552);
        if ( (_DWORD)v29 != -1 )
          goto LABEL_55;
LABEL_59:
        i = v94;
        v23 = 1;
        goto LABEL_60;
      }
      while ( i )
      {
        v29 = *(int *)(v28 + 552);
        --i;
        if ( (_DWORD)v29 == -1 )
          goto LABEL_59;
LABEL_55:
        v28 = qword_4F04C68[0] + 776 * v29;
        if ( !v28 )
          goto LABEL_59;
        v9 = *(unsigned __int8 *)(v28 + 4);
        if ( (_BYTE)v9 != 17 )
          goto LABEL_54;
      }
      v30 = *(_QWORD **)(*(_QWORD *)(v28 + 184) + 40LL);
      if ( !v30 )
      {
LABEL_167:
        v94 = 0;
        v23 = 1;
        goto LABEL_60;
      }
      while ( v27 != *(_DWORD *)(v30[16] + 36LL) )
      {
        v30 = (_QWORD *)v30[14];
        if ( !v30 )
          goto LABEL_167;
      }
      v31 = v30;
      do
      {
        v31 = (_QWORD *)v31[14];
        ++i;
        if ( !v31 )
          break;
        v9 = v31[16];
      }
      while ( v27 == *(_DWORD *)(v9 + 36) );
      v94 = i;
      if ( !*v30 )
      {
        i = v94;
LABEL_145:
        v23 = 0;
LABEL_60:
        m128i_i64[5] = 0;
        goto LABEL_50;
      }
      m128i_i64[10] = v30;
      v32 = *v30;
      v23 = 0;
      m128i_i64[5] = *v30;
      *(_QWORD *)(v32 + 88) = v30;
      i = v94;
    }
    else if ( v13 )
    {
      if ( v13 == 3 )
      {
        v43 = *(const char **)(*(_QWORD *)k->m128i_i64[1] + 8LL);
        v44 = qword_4F04C68[0];
        v45 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( !v45 )
        {
          v94 = 0;
          i = 0;
          v23 = 1;
          goto LABEL_60;
        }
        v83 = 0;
        v80 = m128i_i64;
        do
        {
          v46 = *(_QWORD *)(v45 + 208);
          if ( (unsigned __int8)(*(_BYTE *)(v45 + 4) - 6) <= 1u )
          {
            if ( v46 )
            {
              if ( (*(_BYTE *)(*(_QWORD *)(v46 + 168) + 109LL) & 0x20) != 0 )
              {
                v50 = *(_QWORD *)(v46 + 160);
                if ( v50 )
                {
                  while ( 1 )
                  {
                    v51 = *(const char **)(v50 + 8);
                    if ( v51 )
                    {
                      v9 = (__int64)v43;
                      if ( !strcmp(v43, v51) )
                        break;
                    }
                    v50 = *(_QWORD *)(v50 + 112);
                    if ( !v50 )
                      goto LABEL_106;
                  }
                  v83 = (_QWORD *)v50;
                }
              }
            }
          }
LABEL_106:
          v47 = *(_DWORD *)(v45 + 552);
          if ( v47 == -1 )
            break;
          v45 = v44 + 776LL * v47;
        }
        while ( v45 );
        v48 = v83;
        m128i_i64 = v80;
        v94 = 0;
        if ( !v83 )
        {
          i = 0;
          v23 = 1;
          goto LABEL_60;
        }
        for ( i = 1; ; ++i )
        {
          v48 = (_QWORD *)v48[14];
          if ( !v48 || v83[1] != v48[1] )
            break;
        }
        v94 = i;
        v49 = *v83;
        if ( !*v83 )
          goto LABEL_145;
        v80[5] = v49;
        v80[10] = v83;
        *(_QWORD *)(v49 + 88) = v83;
        v23 = 0;
      }
      else if ( v13 == 4 )
      {
        v9 = a2;
        LODWORD(v96) = 0;
        v52 = sub_85BA20(a2, a3, k->m128i_i64[1], &v94, v99, (int *)&v96);
        v53 = k[6].m128i_i8[1];
        v94 = 0;
        v54 = *(_QWORD *)(v52 + 32);
        for ( j = *(_BYTE *)(v54 + 140); j == 12; j = *(_BYTE *)(v54 + 140) )
          v54 = *(_QWORD *)(v54 + 160);
        if ( (unsigned __int8)(j - 9) > 2u )
        {
          i = 0;
          v56 = 0;
        }
        else if ( **(_QWORD **)(v54 + 168) )
        {
          v56 = 0;
          v81 = m128i_i64;
          v57 = 0;
          v58 = **(__int64 ****)(v54 + 168);
          v59 = v53 == 0;
          do
          {
            if ( ((_BYTE)v58[12] & 1) != 0 || v59 )
            {
              v9 = 0;
              v60 = sub_725090(0);
              v61 = v58[5];
              *((_BYTE *)v60 + 24) |= 8u;
              v60[4] = v61;
              if ( !v56 )
                v56 = v60;
              if ( v57 )
                *v57 = v60;
              ++v94;
              v57 = v60;
            }
            v58 = (__int64 **)*v58;
          }
          while ( v58 );
          m128i_i64 = v81;
          i = v94;
        }
        else
        {
          v56 = 0;
          i = 0;
        }
        m128i_i64[10] = v56;
        v23 = 0;
        m128i_i64[9] = v99[0];
      }
      else if ( a4 )
      {
        if ( a7 )
        {
          v94 = 0;
          v14 = *(_QWORD **)a7;
          if ( *(_QWORD *)a7 )
          {
            v79 = m128i_i64;
            v15 = 0;
            v16 = 0;
            v17 = (_QWORD *)k->m128i_i64[1];
            v82 = k;
            v18 = *(_DWORD *)(v17[11] + 120LL);
            do
            {
              while ( 1 )
              {
                v19 = v14[2];
                if ( v18 == *(_DWORD *)(v19 + 36) )
                {
                  v9 = *(_QWORD *)(v19 + 24);
                  if ( !strcmp((const char *)v9, *(const char **)(*v17 + 8LL)) )
                    break;
                }
                v14 = (_QWORD *)*v14;
                if ( !v14 )
                  goto LABEL_22;
              }
              v15 = (__int64 *)v14[1];
              v16 = v14;
              v14 = (_QWORD *)*v14;
            }
            while ( v14 );
LABEL_22:
            v20 = v18;
            m128i_i64 = v79;
            for ( k = v82; v15; v15 = (__int64 *)*v15 )
            {
              if ( (*((_BYTE *)v15 + 33) & 2) == 0 )
                break;
              if ( v20 != *((_DWORD *)v15 + 9) )
                break;
              ++v94;
            }
            if ( v16 )
            {
              v21 = v16[1];
              if ( v21 && (*(_BYTE *)(v21 + 33) & 1) != 0 )
              {
                v95 = 0;
                sub_89F7D0(&v96);
                v62 = v98;
                if ( v98 == v97 )
                  sub_738390(&v96);
                v63 = &v96->m128i_i8[24 * v62];
                if ( v63 )
                {
                  v63[16] &= 0xF0u;
                  *(_QWORD *)v63 = a2;
                  *((_QWORD *)v63 + 1) = a3;
                }
                v99[0] = 0;
                v98 = v62 + 1;
                v64 = _mm_loadu_si128((const __m128i *)(a7 + 16));
                v65 = _mm_loadu_si128((const __m128i *)(a7 + 32));
                v99[1] = 0;
                v66 = _mm_loadu_si128((const __m128i *)(a7 + 48));
                v67 = _mm_loadu_si128((const __m128i *)(a7 + 64));
                v100 = v64;
                v68 = _mm_loadu_si128((const __m128i *)(a7 + 80));
                v101 = v65;
                v104 = v68;
                v102 = v66;
                v103 = v67;
                v104.m128i_i32[3] = 0;
                v69 = (__m128i *)sub_724EF0(*(_QWORD *)(v16[1] + 8LL));
                v70 = (const __m128i *)v16[1];
                *v69 = _mm_loadu_si128(v70);
                v69[1] = _mm_loadu_si128(v70 + 1);
                v69[2] = _mm_loadu_si128(v70 + 2);
                v69[3] = _mm_loadu_si128(v70 + 3);
                v69[4] = _mm_loadu_si128(v70 + 4);
                v71 = v70[5].m128i_i64[0];
                v69->m128i_i64[0] = 0;
                v69[5].m128i_i64[0] = v71;
                v72 = sub_8A4D00(v69, &v96, (char *)v82[1].m128i_i64 + 4, 0, &v95, v99);
                sub_724F80(v69->m128i_i64);
                sub_8921C0(v99[0]);
                v94 = 0;
                if ( v104.m128i_i32[3] )
                {
                  sub_724F80((__int64 *)v72);
                  *(_DWORD *)(a7 + 92) = 1;
                }
                else
                {
                  v77 = v95;
                  v16[1] = v72;
                  if ( !v77 )
                  {
                    while ( v72 && (*(_BYTE *)(v72 + 33) & 2) != 0 && *(_DWORD *)(v72 + 36) == v82[1].m128i_i32[0] )
                    {
                      ++v94;
                      v72 = *(_QWORD *)v72;
                    }
                  }
                }
                v9 = (__int64)v96;
                sub_823A00((__int64)v96, 24 * v97, v73, v74, v75, v76);
                v21 = v16[1];
              }
              v79[10] = v21;
              i = v94;
              v23 = 0;
LABEL_30:
              m128i_i64[7] = v16;
              if ( v91 )
                goto LABEL_51;
              goto LABEL_31;
            }
            i = v94;
          }
          else
          {
            i = 0;
          }
          m128i_i64[10] = 0;
          v23 = a4;
          v16 = 0;
          goto LABEL_30;
        }
        i = v94;
        v23 = 0;
      }
      else
      {
        v39 = dword_4F04C64;
        while ( 1 )
        {
          v40 = qword_4F04C68[0] + 776 * v39;
          if ( !v40 || *(_BYTE *)(v40 + 4) != 1 )
          {
LABEL_101:
            v94 = 0;
            v23 = 0;
            i = 0;
            m128i_i64[5] = 0;
            goto LABEL_50;
          }
          v41 = *(__int64 **)(v40 + 752);
          if ( v41 )
            break;
LABEL_100:
          v39 = *(int *)(v40 + 552);
          if ( (_DWORD)v39 == -1 )
            goto LABEL_101;
        }
        while ( k[1].m128i_i32[0] != *((_DWORD *)v41 + 30) )
        {
          v41 = (__int64 *)*v41;
          if ( !v41 )
            goto LABEL_100;
        }
        v94 = 0;
        v42 = v41;
        i = 0;
        do
        {
          if ( (*((_BYTE *)v42 + 42) & 2) == 0 )
            break;
          v94 = ++i;
          v42 = (__int64 *)*v42;
        }
        while ( v42 );
        m128i_i64[10] = v41;
        v23 = 0;
        m128i_i64[5] = v41[1];
      }
    }
    else
    {
      v9 = a2;
      LODWORD(v96) = 0;
      v26 = sub_85BA20(a2, a3, k->m128i_i64[1], &v94, v99, (int *)&v96);
      v23 = v26 == 0;
      if ( !(_DWORD)v96 && a7 )
      {
        v9 = a7;
        *(_DWORD *)(a7 + 92) = 1;
      }
      m128i_i64[10] = v26;
      i = v94;
      m128i_i64[9] = v99[0];
    }
LABEL_50:
    if ( v91 )
    {
LABEL_51:
      v91 = 0;
      v90 = v94;
      goto LABEL_5;
    }
LABEL_31:
    if ( v90 == i )
      goto LABEL_5;
    if ( (unsigned int)sub_85B1E0(v9, v23) )
    {
      v87 = 1;
      goto LABEL_5;
    }
    if ( a4 )
      break;
    sub_686610(
      0x781u,
      &k[1].m128i_i32[1],
      *(_QWORD *)(*(_QWORD *)k->m128i_i64[1] + 8LL),
      *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 8LL) + 8LL));
    v89 = 1;
LABEL_5:
    k = (const __m128i *)k->m128i_i64[0];
    if ( !k )
      goto LABEL_38;
LABEL_6:
    v94 = 0;
    v9 = (__int64)k;
    v12 = sub_8665B0(k);
    if ( v93 )
      *m128i_i64 = v12;
    else
      v93 = v12;
    m128i_i64 = v12->m128i_i64;
  }
  v89 = 0;
  if ( !*(_DWORD *)(a7 + 80) && (v89 = a4, (_DWORD)v23) )
  {
    v91 = 0;
    v89 = *(_DWORD *)(a7 + 84) == 0;
  }
  else
  {
    v91 = 0;
  }
  k = (const __m128i *)k->m128i_i64[0];
  if ( k )
    goto LABEL_6;
LABEL_38:
  if ( a5 )
  {
    v37 = v87;
    v38 = v90 == 0;
LABEL_138:
    v8 = (const __m128i *)qword_4F5FD30;
    if ( qword_4F5FD30 )
      qword_4F5FD30 = *(_QWORD *)qword_4F5FD30;
    else
      v8 = (const __m128i *)sub_823970(24);
    v8->m128i_i64[0] = 0;
    v8[1].m128i_i8[0] = 0;
    v8->m128i_i64[1] = (__int64)v93;
    v8[1].m128i_i8[2] = v37;
    v8[1].m128i_i8[1] = v38 && a6 != 0;
    goto LABEL_44;
  }
  if ( !v89 && v90 | a6 )
  {
    v37 = v87;
    v38 = v90 == 0;
    goto LABEL_138;
  }
  v8 = 0;
  if ( v93 )
  {
    v8 = v93;
    do
    {
      v24 = (__int64 *)v8;
      v8 = (const __m128i *)v8->m128i_i64[0];
    }
    while ( v8 );
    *v24 = qword_4F5FD48;
    qword_4F5FD48 = (__int64)v93;
  }
LABEL_44:
  *a8 = v89;
  return v8;
}
