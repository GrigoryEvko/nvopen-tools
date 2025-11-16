// Function: sub_28654E0
// Address: 0x28654e0
//
int __fastcall sub_28654E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r10
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r10
  __int64 v15; // r12
  __int64 v16; // r15
  __int64 v17; // r13
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rbx
  size_t v20; // r15
  __int64 v21; // r12
  const __m128i *v22; // rax
  __int64 v23; // r15
  __int64 v24; // r13
  __int64 v25; // r12
  const __m128i *v26; // rbx
  __int64 v27; // rdx
  unsigned __int64 v28; // rsi
  const __m128i *v29; // r15
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdx
  __m128i *v32; // r14
  __int64 v33; // r14
  __int64 v34; // r15
  __int64 *v35; // r13
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // r14
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rdi
  __int64 v44; // r14
  unsigned __int64 v45; // r12
  __int64 v46; // rax
  unsigned __int64 *v47; // r14
  unsigned __int64 *v48; // r12
  unsigned __int64 v49; // r14
  __int8 *v50; // r15
  __int64 v51; // r15
  _QWORD *v52; // rax
  _QWORD *v53; // r15
  unsigned __int64 v54; // rsi
  _QWORD *v55; // r13
  unsigned __int64 *v56; // rdi
  unsigned __int64 v57; // rcx
  __int64 v58; // r10
  __int64 v59; // rdx
  unsigned __int64 v60; // rcx
  __int64 v61; // rcx
  __int64 v63; // [rsp-10h] [rbp-80h]
  unsigned __int64 v64; // [rsp-8h] [rbp-78h]
  __int64 v65; // [rsp+8h] [rbp-68h]
  __int64 v66; // [rsp+10h] [rbp-60h]
  __int64 v67; // [rsp+10h] [rbp-60h]
  __int64 v68; // [rsp+18h] [rbp-58h]
  __int64 v69; // [rsp+18h] [rbp-58h]
  __int64 v70; // [rsp+18h] [rbp-58h]
  unsigned __int64 v71; // [rsp+20h] [rbp-50h]
  __int64 v72; // [rsp+28h] [rbp-48h]
  char v73; // [rsp+28h] [rbp-48h]
  __int64 v74; // [rsp+28h] [rbp-48h]
  __int64 v75; // [rsp+30h] [rbp-40h]
  __int64 v76; // [rsp+30h] [rbp-40h]
  const __m128i *v77; // [rsp+30h] [rbp-40h]
  char v78; // [rsp+30h] [rbp-40h]
  unsigned int v79; // [rsp+30h] [rbp-40h]
  unsigned __int64 v80; // [rsp+38h] [rbp-38h]

  v3 = a1;
  v4 = *(unsigned int *)(a1 + 1328);
  v5 = *(_QWORD *)(a1 + 1320);
  v6 = (unsigned int)qword_5001308;
  v71 = v4;
  v7 = v5 + 2184 * v4;
  if ( v5 == v7 )
  {
    if ( (unsigned int)qword_5001308 > 1 )
      return v5;
  }
  else
  {
    a3 = *(_QWORD *)(a1 + 1320);
    v4 = 1;
    while ( 1 )
    {
      v8 = *(unsigned int *)(a3 + 768);
      if ( (unsigned int)v8 >= (unsigned int)qword_5001308 )
        break;
      v4 *= v8;
      if ( (unsigned int)qword_5001308 <= v4 )
        break;
      a3 += 2184;
      if ( v7 == a3 )
        return v5;
    }
  }
  if ( !v71 )
    return v5;
  v80 = 0;
  while ( 2 )
  {
    v9 = v5 + 2184 * v80;
    v5 = *(unsigned int *)(v9 + 768);
    v10 = *(_QWORD *)(v9 + 760);
    v11 = v10 + 112 * v5;
    if ( v10 == v11 )
    {
      ++v80;
      goto LABEL_55;
    }
    v5 = v3;
    v12 = v11;
    v13 = *(_QWORD *)(v9 + 760);
    v14 = v5;
    while ( 1 )
    {
      if ( *(_QWORD *)(v13 + 8) )
      {
        if ( *(_QWORD *)(v13 + 32) <= 1u )
        {
          v15 = *(_QWORD *)(v14 + 1320);
          v16 = v15 + 2184LL * *(unsigned int *)(v14 + 1328);
          if ( v15 != v16 )
          {
            while ( 1 )
            {
              if ( v9 != v15 )
              {
                LODWORD(v5) = *(_DWORD *)(v15 + 32);
                if ( (_DWORD)v5 != 3 && (_DWORD)v5 == *(_DWORD *)(v9 + 32) )
                {
                  v5 = *(_QWORD *)(v9 + 40);
                  if ( *(_QWORD *)(v15 + 40) == v5 )
                  {
                    LODWORD(v5) = *(_DWORD *)(v9 + 48);
                    if ( *(_DWORD *)(v15 + 48) == (_DWORD)v5 )
                    {
                      v5 = *(_QWORD *)(v9 + 752);
                      if ( *(_QWORD *)(v15 + 752) == v5 )
                      {
                        v75 = v14;
                        LODWORD(v5) = sub_2864E10(v15, v13, a3, v4, v7, v6);
                        v14 = v75;
                        if ( (_BYTE)v5 )
                        {
                          v5 = *(unsigned int *)(v15 + 768);
                          v4 = *(_QWORD *)(v15 + 760);
                          v7 = v4 + 112 * v5;
                          if ( v4 != v7 )
                          {
                            v6 = *(unsigned int *)(v13 + 48);
                            v76 = v9;
                            v17 = v13;
                            v18 = *(_QWORD *)(v15 + 760);
                            v72 = v12;
                            v19 = v4 + 112 * v5;
                            a3 = 8 * v6;
                            v68 = v16;
                            v66 = v15;
                            v20 = 8 * v6;
                            v21 = v6;
                            v65 = v14;
                            while ( 1 )
                            {
                              v5 = *(unsigned int *)(v18 + 48);
                              if ( v5 == v21 )
                              {
                                if ( !v20
                                  || (LODWORD(v5) = memcmp(*(const void **)(v18 + 40), *(const void **)(v17 + 40), v20),
                                      !(_DWORD)v5) )
                                {
                                  v5 = *(_QWORD *)(v17 + 88);
                                  if ( *(_QWORD *)(v18 + 88) == v5 )
                                  {
                                    v5 = *(_QWORD *)v17;
                                    if ( *(_QWORD *)v18 == *(_QWORD *)v17 )
                                    {
                                      v5 = *(_QWORD *)(v17 + 32);
                                      if ( *(_QWORD *)(v18 + 32) == v5 )
                                      {
                                        v5 = *(_QWORD *)(v17 + 96);
                                        if ( *(_QWORD *)(v18 + 96) == v5 )
                                        {
                                          LODWORD(v5) = *(unsigned __int8 *)(v17 + 104);
                                          if ( *(_BYTE *)(v18 + 104) == (_BYTE)v5 )
                                            break;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                              v18 += 112LL;
                              if ( v19 == v18 )
                              {
                                v13 = v17;
                                v12 = v72;
                                v16 = v68;
                                v9 = v76;
                                v15 = v66;
                                v14 = v65;
                                goto LABEL_15;
                              }
                            }
                            v4 = v18;
                            v12 = v72;
                            v13 = v17;
                            v16 = v68;
                            v15 = v66;
                            v9 = v76;
                            v14 = v65;
                            if ( !*(_QWORD *)(v4 + 8) )
                              break;
                          }
                        }
                      }
                    }
                  }
                }
              }
LABEL_15:
              v15 += 2184;
              if ( v16 == v15 )
                goto LABEL_10;
            }
            LODWORD(v5) = sub_2850900(
                            v65,
                            v66,
                            *(_QWORD *)(v13 + 8),
                            *(_BYTE *)(v13 + 16),
                            0,
                            *(_DWORD *)(v76 + 32),
                            *(_QWORD ***)(v76 + 40),
                            *(_QWORD *)(v76 + 48));
            a3 = v63;
            v14 = v65;
            v4 = v64;
            if ( (_BYTE)v5 )
              break;
          }
        }
      }
LABEL_10:
      v13 += 112;
      if ( v12 == v13 )
      {
        ++v80;
        v3 = v14;
        goto LABEL_55;
      }
    }
    v73 = v5;
    v3 = v65;
    *(_BYTE *)(v66 + 744) &= *(_BYTE *)(v76 + 744);
    v22 = *(const __m128i **)(v76 + 56);
    v69 = v66 + 56;
    v23 = 5LL * *(unsigned int *)(v76 + 64);
    v77 = &v22[v23];
    if ( &v22[v23] == v22 )
      goto LABEL_62;
    v67 = v9;
    v24 = v15;
    v25 = v13;
    v26 = v22;
    do
    {
      v26[4].m128i_i64[0] += *(_QWORD *)(v25 + 8);
      if ( *(_QWORD *)(v25 + 8) )
        v26[4].m128i_i8[8] = *(_BYTE *)(v25 + 16);
      v28 = *(unsigned int *)(v24 + 64);
      v29 = v26;
      v30 = *(_QWORD *)(v24 + 56);
      v6 = v28 + 1;
      v31 = v28;
      if ( v28 + 1 > *(unsigned int *)(v24 + 68) )
      {
        if ( v30 > (unsigned __int64)v26 || (v31 = v30 + 80 * v28, v31 <= (unsigned __int64)v26) )
        {
          v29 = v26;
          sub_2851200(v69, v28 + 1, v31, v4, v7, v6);
          v28 = *(unsigned int *)(v24 + 64);
          v30 = *(_QWORD *)(v24 + 56);
          LODWORD(v31) = *(_DWORD *)(v24 + 64);
        }
        else
        {
          v50 = &v26->m128i_i8[-v30];
          sub_2851200(v69, v28 + 1, v31, v4, v7, v6);
          v30 = *(_QWORD *)(v24 + 56);
          v28 = *(unsigned int *)(v24 + 64);
          v29 = (const __m128i *)&v50[v30];
          LODWORD(v31) = *(_DWORD *)(v24 + 64);
        }
      }
      v32 = (__m128i *)(v30 + 80 * v28);
      if ( v32 )
      {
        v32->m128i_i64[0] = v29->m128i_i64[0];
        v32->m128i_i64[1] = v29->m128i_i64[1];
        sub_C8CD80((__int64)v32[1].m128i_i64, (__int64)v32[3].m128i_i64, (__int64)v29[1].m128i_i64, v4, v7, v6);
        v32[4] = _mm_loadu_si128(v29 + 4);
        LODWORD(v31) = *(_DWORD *)(v24 + 64);
      }
      *(_DWORD *)(v24 + 64) = v31 + 1;
      if ( v26[4].m128i_i8[8] )
      {
        if ( v26[4].m128i_i64[0] > *(_QWORD *)(v24 + 728) )
        {
LABEL_40:
          *(_QWORD *)(v24 + 728) = v26[4].m128i_i64[0];
          *(_BYTE *)(v24 + 736) = v26[4].m128i_i8[8];
          if ( !v26[4].m128i_i8[8] )
          {
LABEL_42:
            v27 = v26[4].m128i_i64[0];
            goto LABEL_43;
          }
        }
        if ( !*(_BYTE *)(v24 + 720) )
          goto LABEL_45;
        goto LABEL_42;
      }
      v27 = v26[4].m128i_i64[0];
      if ( !*(_BYTE *)(v24 + 736) && *(_QWORD *)(v24 + 728) < v27 )
        goto LABEL_40;
LABEL_43:
      if ( *(_QWORD *)(v24 + 712) > v27 )
      {
        *(_QWORD *)(v24 + 712) = v26[4].m128i_i64[0];
        *(_BYTE *)(v24 + 720) = v26[4].m128i_i8[8];
      }
LABEL_45:
      v26 += 5;
    }
    while ( v26 != v77 );
    v15 = v24;
    v3 = v65;
    v9 = v67;
LABEL_62:
    v33 = *(unsigned int *)(v15 + 768);
    if ( *(_DWORD *)(v15 + 768) )
    {
      v78 = 0;
      v34 = 0;
      v70 = v9;
      do
      {
        while ( 1 )
        {
          v35 = (__int64 *)(*(_QWORD *)(v15 + 760) + 112 * v34);
          if ( !sub_2850770(
                  *(__int64 **)(v3 + 48),
                  *(_QWORD *)(v15 + 712),
                  *(_BYTE *)(v15 + 720),
                  *(_QWORD *)(v15 + 728),
                  *(_BYTE *)(v15 + 736),
                  *(_DWORD *)(v15 + 32),
                  *(_QWORD *)(v15 + 40),
                  *(_DWORD *)(v15 + 48),
                  (__int64)v35) )
            break;
          if ( v33 == ++v34 )
            goto LABEL_67;
        }
        --v33;
        sub_28532A0(v15, v35);
        v78 = v73;
      }
      while ( v33 != v34 );
LABEL_67:
      v9 = v70;
      if ( v78 )
        sub_2855860(v15, -251719695 * ((v15 - *(_QWORD *)(v3 + 1320)) >> 3), v3 + 36280);
    }
    v36 = *(_QWORD *)(v3 + 1320);
    v37 = *(unsigned int *)(v3 + 1328);
    v38 = v36 + 2184 * v37 - 2184;
    if ( v9 != v38 )
    {
      sub_28574C0(v9, v38, 2184 * v37, v36, v7, v6);
      LODWORD(v37) = *(_DWORD *)(v3 + 1328);
      v36 = *(_QWORD *)(v3 + 1320);
    }
    v39 = (unsigned int)(v37 - 1);
    *(_DWORD *)(v3 + 1328) = v39;
    v40 = v36 + 2184 * v39;
    if ( !*(_BYTE *)(v40 + 2148) )
      _libc_free(*(_QWORD *)(v40 + 2128));
    v41 = *(_QWORD *)(v40 + 760);
    v42 = v41 + 112LL * *(unsigned int *)(v40 + 768);
    if ( v41 != v42 )
    {
      do
      {
        v42 -= 112LL;
        v43 = *(_QWORD *)(v42 + 40);
        if ( v43 != v42 + 56 )
          _libc_free(v43);
      }
      while ( v41 != v42 );
      v42 = *(_QWORD *)(v40 + 760);
    }
    if ( v42 != v40 + 776 )
      _libc_free(v42);
    v44 = *(_QWORD *)(v40 + 56);
    v45 = v44 + 80LL * *(unsigned int *)(v40 + 64);
    if ( v44 != v45 )
    {
      do
      {
        while ( 1 )
        {
          v45 -= 80LL;
          if ( !*(_BYTE *)(v45 + 44) )
            break;
          if ( v44 == v45 )
            goto LABEL_85;
        }
        _libc_free(*(_QWORD *)(v45 + 24));
      }
      while ( v44 != v45 );
LABEL_85:
      v45 = *(_QWORD *)(v40 + 56);
    }
    if ( v45 != v40 + 72 )
      _libc_free(v45);
    v46 = *(unsigned int *)(v40 + 24);
    if ( (_DWORD)v46 )
    {
      v47 = *(unsigned __int64 **)(v40 + 8);
      v48 = &v47[6 * v46];
      do
      {
        if ( (unsigned __int64 *)*v47 != v47 + 2 )
          _libc_free(*v47);
        v47 += 6;
      }
      while ( v48 != v47 );
      v46 = *(unsigned int *)(v40 + 24);
    }
    LODWORD(v5) = sub_C7D6A0(*(_QWORD *)(v40 + 8), 48 * v46, 8);
    v49 = *(unsigned int *)(v3 + 1328);
    v79 = *(_DWORD *)(v3 + 1328);
    if ( *(_DWORD *)(v3 + 36296) )
    {
      a3 = *(_QWORD *)(v3 + 36288);
      v7 = 16LL * *(unsigned int *)(v3 + 36304);
      v51 = a3 + v7;
      if ( a3 != a3 + v7 )
      {
        while ( 1 )
        {
          v5 = *(_QWORD *)a3;
          if ( *(_QWORD *)a3 != -8192 && v5 != -4096 )
            break;
          a3 += 16;
          if ( v51 == a3 )
            goto LABEL_95;
        }
        if ( a3 != v51 )
        {
          v52 = (_QWORD *)v51;
          v53 = (_QWORD *)a3;
          v54 = *(_QWORD *)(a3 + 8);
          v55 = v52;
          v74 = 1LL << v80;
          v56 = (unsigned __int64 *)(a3 + 8);
          if ( (v54 & 1) != 0 )
          {
LABEL_107:
            v57 = v54 >> 58;
            if ( v54 >> 58 <= v80 )
              goto LABEL_113;
            if ( v49 < v57 && (v58 = ~(-1LL << v57), v59 = v58 & (v54 >> 1), _bittest64(&v59, v79)) )
            {
              v60 = 2 * ((v57 << 57) | v58 & (v74 | v59)) + 1;
              v53[1] = v60;
            }
            else
            {
              v60 = 2 * ((v54 >> 58 << 57) | ~(-1LL << (v54 >> 58)) & ~v74 & (v54 >> 1)) + 1;
              v53[1] = v60;
            }
LABEL_111:
            if ( (v60 & 1) != 0 )
              goto LABEL_112;
          }
          else
          {
            while ( 1 )
            {
              v57 = *(unsigned int *)(v54 + 64);
              if ( v57 > v80 )
                break;
LABEL_113:
              if ( v49 <= v57 )
                v57 = v49;
              v53 += 2;
              LODWORD(v5) = sub_228BF90(v56, v57, 0, v57, v7, v6);
              if ( v53 == v55 )
                goto LABEL_95;
              while ( 1 )
              {
                v5 = *v53;
                if ( *v53 != -4096 && v5 != -8192 )
                  break;
                v53 += 2;
                if ( v55 == v53 )
                  goto LABEL_95;
              }
              if ( v53 == v55 )
                goto LABEL_95;
              v54 = v53[1];
              v56 = v53 + 1;
              if ( (v54 & 1) != 0 )
                goto LABEL_107;
            }
            if ( v49 < v57 )
            {
              v61 = *(_QWORD *)(*(_QWORD *)v54 + 8LL * (v79 >> 6));
              if ( _bittest64(&v61, v79) )
              {
                *(_QWORD *)(*(_QWORD *)v54 + 8LL * ((unsigned int)v80 >> 6)) |= 1LL << v80;
                v60 = v53[1];
                goto LABEL_111;
              }
            }
            *(_QWORD *)(8LL * ((unsigned int)v80 >> 6) + *(_QWORD *)v54) &= ~(1LL << v80);
            v60 = v53[1];
            if ( (v60 & 1) != 0 )
            {
LABEL_112:
              v57 = v60 >> 58;
              goto LABEL_113;
            }
          }
          v57 = *(unsigned int *)(v60 + 64);
          goto LABEL_113;
        }
      }
    }
LABEL_95:
    --v71;
LABEL_55:
    v4 = v80;
    if ( v71 != v80 )
    {
      v5 = *(_QWORD *)(v3 + 1320);
      continue;
    }
    return v5;
  }
}
