// Function: sub_29B2FB0
// Address: 0x29b2fb0
//
__int64 *__fastcall sub_29B2FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 *v5; // r14
  __int64 v6; // r12
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 *v11; // r12
  unsigned __int64 v12; // rax
  __int64 *result; // rax
  int v15; // edx
  unsigned int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rbx
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // r13
  _BYTE *v23; // rsi
  __int64 v24; // r12
  __int64 v25; // r13
  __int64 v26; // r14
  int v27; // eax
  __int64 v28; // rcx
  int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // rdi
  __int64 i; // r14
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // r12
  int v40; // eax
  __int64 v41; // rcx
  int v42; // edx
  unsigned int v43; // eax
  __int64 v44; // rdi
  __int64 v45; // r13
  const __m128i *v46; // r14
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rdx
  __m128i *v54; // rax
  __int64 v55; // r14
  __int64 v56; // r12
  __int64 v57; // rbx
  const char *v58; // rax
  __int64 *v59; // r12
  __int64 v60; // rcx
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rdx
  int v64; // r8d
  int v65; // r8d
  int v66; // r9d
  __int64 v67; // r13
  const char *v68; // [rsp+10h] [rbp-160h]
  __int64 v69; // [rsp+20h] [rbp-150h]
  __int64 v71; // [rsp+30h] [rbp-140h]
  __int64 v75; // [rsp+58h] [rbp-118h]
  unsigned __int8 *v76; // [rsp+60h] [rbp-110h]
  __int64 *v77; // [rsp+60h] [rbp-110h]
  __int64 v78; // [rsp+68h] [rbp-108h]
  __int64 *v79; // [rsp+78h] [rbp-F8h]
  char v80; // [rsp+80h] [rbp-F0h] BYREF
  char v81; // [rsp+81h] [rbp-EFh]
  __int64 v82; // [rsp+88h] [rbp-E8h]
  __int64 v83; // [rsp+90h] [rbp-E0h]
  __int64 v84[4]; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 *v85; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+C8h] [rbp-A8h]
  _BYTE v87[16]; // [rsp+D0h] [rbp-A0h] BYREF
  _BYTE *v88; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v89; // [rsp+E8h] [rbp-88h]
  _BYTE v90[16]; // [rsp+F0h] [rbp-80h] BYREF
  const char *v91; // [rsp+100h] [rbp-70h] BYREF
  __int64 v92; // [rsp+108h] [rbp-68h]
  _BYTE v93[96]; // [rsp+110h] [rbp-60h] BYREF

  v5 = *(__int64 **)(a1 + 88);
  v6 = 8LL * *(unsigned int *)(a1 + 96);
  v7 = &v5[(unsigned __int64)v6 / 8];
  v8 = *(_QWORD *)(*v5 + 72);
  v88 = 0;
  v75 = v8;
  v71 = a1 + 56;
  v91 = (const char *)(a1 + 56);
  v92 = (__int64)&v88;
  v9 = v6 >> 3;
  v10 = v6 >> 5;
  if ( v10 )
  {
    v11 = &v5[4 * v10];
    while ( !(unsigned __int8)sub_29AABA0(&v91, *v5) )
    {
      if ( (unsigned __int8)sub_29AABA0(&v91, v5[1]) )
      {
        if ( v7 == v5 + 1 )
          goto LABEL_9;
        goto LABEL_103;
      }
      if ( (unsigned __int8)sub_29AABA0(&v91, v5[2]) )
      {
        v5 += 2;
        goto LABEL_8;
      }
      if ( (unsigned __int8)sub_29AABA0(&v91, v5[3]) )
      {
        v5 += 3;
        goto LABEL_8;
      }
      v5 += 4;
      if ( v11 == v5 )
      {
        v9 = v7 - v5;
        goto LABEL_93;
      }
    }
    goto LABEL_8;
  }
LABEL_93:
  if ( v9 == 2 )
    goto LABEL_100;
  if ( v9 == 3 )
  {
    if ( (unsigned __int8)sub_29AABA0(&v91, *v5) )
      goto LABEL_8;
    ++v5;
LABEL_100:
    if ( (unsigned __int8)sub_29AABA0(&v91, *v5) )
      goto LABEL_8;
    ++v5;
    goto LABEL_96;
  }
  if ( v9 != 1 )
    goto LABEL_9;
LABEL_96:
  if ( !(unsigned __int8)sub_29AABA0(&v91, *v5) )
    goto LABEL_9;
LABEL_8:
  if ( v7 == v5 )
  {
LABEL_9:
    v12 = (unsigned __int64)v88;
    goto LABEL_10;
  }
LABEL_103:
  v12 = 0;
LABEL_10:
  *a5 = v12;
  result = *(__int64 **)a2;
  v79 = *(__int64 **)a2;
  v78 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v78 )
  {
    do
    {
      v18 = *(_QWORD *)(a1 + 64);
      v19 = *v79;
      v20 = *(_DWORD *)(a1 + 80);
      v21 = *(_QWORD *)(*v79 + 40);
      if ( v20 )
      {
        v15 = v20 - 1;
        v16 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v17 = *(_QWORD *)(v18 + 8LL * v16);
        if ( v21 == v17 )
          goto LABEL_13;
        v64 = 1;
        while ( v17 != -4096 )
        {
          v16 = v15 & (v64 + v16);
          v17 = *(_QWORD *)(v18 + 8LL * v16);
          if ( v21 == v17 )
            goto LABEL_13;
          ++v64;
        }
      }
      v22 = *(_QWORD *)(v21 + 72);
      if ( v22 == v75 )
      {
        v23 = (_BYTE *)a1;
        sub_29AAFB0((__int64)&v80, a1, a2, *v79, *a5);
        if ( !v82 )
        {
          v24 = *(_QWORD *)(v19 + 16);
          v85 = (__int64 *)v87;
          v86 = 0x200000000LL;
          if ( v24 )
          {
            v69 = v22;
            v25 = v24;
            v76 = (unsigned __int8 *)v19;
            do
            {
              v26 = *(_QWORD *)(v25 + 24);
              if ( *(_BYTE *)v26 > 0x1Cu )
              {
                v27 = *(_DWORD *)(a1 + 80);
                v28 = *(_QWORD *)(v26 + 40);
                v23 = *(_BYTE **)(a1 + 64);
                if ( v27 )
                {
                  v29 = v27 - 1;
                  v30 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                  v31 = *(_QWORD *)&v23[8 * v30];
                  if ( v28 == v31 )
                  {
LABEL_27:
                    if ( v76 == sub_BD4070(*(unsigned __int8 **)(v25 + 24), (__int64)v23) )
                    {
                      for ( i = *(_QWORD *)(v26 + 16); i; i = *(_QWORD *)(i + 8) )
                      {
                        v33 = *(_QWORD *)(i + 24);
                        if ( *(_BYTE *)v33 == 85 )
                        {
                          v34 = *(_QWORD *)(v33 - 32);
                          if ( v34 )
                          {
                            if ( !*(_BYTE *)v34
                              && *(_QWORD *)(v34 + 24) == *(_QWORD *)(v33 + 80)
                              && (*(_BYTE *)(v34 + 33) & 0x20) != 0
                              && (unsigned int)(*(_DWORD *)(v34 + 36) - 210) <= 1 )
                            {
                              v23 = *(_BYTE **)(i + 24);
                              if ( !(unsigned __int8)sub_29AACC0(v71, (__int64)v23) )
                              {
                                v37 = (unsigned int)v86;
                                v38 = (unsigned int)v86 + 1LL;
                                if ( v38 > HIDWORD(v86) )
                                {
                                  v23 = v87;
                                  sub_C8D5F0((__int64)&v85, v87, v38, 8u, v35, v36);
                                  v37 = (unsigned int)v86;
                                }
                                v85[v37] = v33;
                                LODWORD(v86) = v86 + 1;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  else
                  {
                    v66 = 1;
                    while ( v31 != -4096 )
                    {
                      v30 = v29 & (v66 + v30);
                      v31 = *(_QWORD *)&v23[8 * v30];
                      if ( v28 == v31 )
                        goto LABEL_27;
                      ++v66;
                    }
                  }
                }
              }
              v25 = *(_QWORD *)(v25 + 8);
            }
            while ( v25 );
            v59 = v85;
            v19 = (__int64)v76;
            v77 = &v85[(unsigned int)v86];
            if ( v77 != v85 )
            {
              do
              {
                v61 = *v59;
                v62 = sub_BCE3C0(**(__int64 ***)(v69 + 40), 0);
                v93[17] = 1;
                v91 = "lt.cast";
                v93[16] = 3;
                v63 = sub_B52210(v19, v62, (__int64)&v91, v61 + 24, 0);
                if ( (*(_BYTE *)(v61 + 7) & 0x40) != 0 )
                  v60 = *(_QWORD *)(v61 - 8);
                else
                  v60 = v61 - 32LL * (*(_DWORD *)(v61 + 4) & 0x7FFFFFF);
                v23 = *(_BYTE **)(v60 + 32);
                ++v59;
                sub_BD2ED0(v61, (__int64)v23, v63);
              }
              while ( v77 != v59 );
            }
          }
          v88 = v90;
          v89 = 0x200000000LL;
          v91 = v93;
          v92 = 0x200000000LL;
          v39 = *(_QWORD *)(v19 + 16);
          if ( v39 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                v45 = *(_QWORD *)(v39 + 24);
                if ( (unsigned __int8 *)v19 != sub_BD4070((unsigned __int8 *)v45, (__int64)v23) )
                  break;
                v23 = (_BYTE *)a1;
                v46 = (const __m128i *)v84;
                sub_29AAFB0((__int64)v84, a1, a2, v45, *a5);
                if ( !v84[1] )
                  break;
                v49 = (unsigned int)v89;
                v50 = (unsigned int)v89 + 1LL;
                if ( v50 > HIDWORD(v89) )
                {
                  v23 = v90;
                  sub_C8D5F0((__int64)&v88, v90, v50, 8u, v47, v48);
                  v49 = (unsigned int)v89;
                }
                *(_QWORD *)&v88[8 * v49] = v45;
                v51 = (unsigned int)v92;
                LODWORD(v89) = v89 + 1;
                v52 = (unsigned int)v92 + 1LL;
                if ( v52 > HIDWORD(v92) )
                {
                  v67 = (__int64)v91;
                  v23 = v93;
                  if ( v91 > (const char *)v84 || v84 >= (__int64 *)&v91[24 * (unsigned int)v92] )
                  {
                    sub_C8D5F0((__int64)&v91, v93, v52, 0x18u, v47, v48);
                    v53 = (__int64)v91;
                    v51 = (unsigned int)v92;
                  }
                  else
                  {
                    sub_C8D5F0((__int64)&v91, v93, v52, 0x18u, v47, v48);
                    v53 = (__int64)v91;
                    v51 = (unsigned int)v92;
                    v46 = (const __m128i *)((char *)v84 + (_QWORD)v91 - v67);
                  }
                }
                else
                {
                  v53 = (__int64)v91;
                }
                v54 = (__m128i *)(v53 + 24 * v51);
                *v54 = _mm_loadu_si128(v46);
                v54[1].m128i_i64[0] = v46[1].m128i_i64[0];
                LODWORD(v92) = v92 + 1;
                v39 = *(_QWORD *)(v39 + 8);
                if ( !v39 )
                {
LABEL_60:
                  if ( (_DWORD)v89 )
                  {
                    v84[0] = v19;
                    sub_29B2110(a3, v84);
                    v55 = 8LL * (unsigned int)v89;
                    v56 = 0;
                    if ( (_DWORD)v89 )
                    {
                      do
                      {
                        while ( 1 )
                        {
                          v57 = *(_QWORD *)&v88[v56];
                          v58 = &v91[3 * v56];
                          if ( *((_QWORD *)v58 + 1) )
                          {
                            if ( *v58 )
                            {
                              v68 = &v91[3 * v56];
                              v84[0] = *((_QWORD *)v58 + 1);
                              sub_29B2110(a3, v84);
                              v58 = v68;
                            }
                            if ( v58[1] )
                            {
                              v84[0] = *((_QWORD *)v58 + 2);
                              sub_29B2110(a4, v84);
                            }
                          }
                          if ( !(unsigned __int8)sub_29AACC0(v71, v57) )
                            break;
                          v56 += 8;
                          if ( v55 == v56 )
                            goto LABEL_48;
                        }
                        v56 += 8;
                        v84[0] = v57;
                        sub_29B2110(a3, v84);
                      }
                      while ( v55 != v56 );
                    }
                  }
                  goto LABEL_48;
                }
              }
              if ( *(_BYTE *)v45 <= 0x1Cu )
                goto LABEL_48;
              v40 = *(_DWORD *)(a1 + 80);
              v41 = *(_QWORD *)(v45 + 40);
              v23 = *(_BYTE **)(a1 + 64);
              if ( !v40 )
                goto LABEL_48;
              v42 = v40 - 1;
              v43 = (v40 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v44 = *(_QWORD *)&v23[8 * v43];
              if ( v41 != v44 )
                break;
LABEL_45:
              v39 = *(_QWORD *)(v39 + 8);
              if ( !v39 )
                goto LABEL_60;
            }
            v65 = 1;
            while ( v44 != -4096 )
            {
              v43 = v42 & (v65 + v43);
              v44 = *(_QWORD *)&v23[8 * v43];
              if ( v41 == v44 )
                goto LABEL_45;
              ++v65;
            }
LABEL_48:
            if ( v91 != v93 )
              _libc_free((unsigned __int64)v91);
          }
          if ( v88 != v90 )
            _libc_free((unsigned __int64)v88);
          if ( v85 != (__int64 *)v87 )
            _libc_free((unsigned __int64)v85);
          goto LABEL_13;
        }
        if ( v80 )
        {
          v91 = (const char *)v82;
          sub_29B2110(a3, (__int64 *)&v91);
          if ( !v81 )
            goto LABEL_19;
        }
        else if ( !v81 )
        {
LABEL_19:
          v91 = (const char *)v19;
          sub_29B2110(a3, (__int64 *)&v91);
          goto LABEL_13;
        }
        v91 = (const char *)v83;
        sub_29B2110(a4, (__int64 *)&v91);
        goto LABEL_19;
      }
LABEL_13:
      result = ++v79;
    }
    while ( (__int64 *)v78 != v79 );
  }
  return result;
}
