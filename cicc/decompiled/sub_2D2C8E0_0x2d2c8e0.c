// Function: sub_2D2C8E0
// Address: 0x2d2c8e0
//
__m128i *__fastcall sub_2D2C8E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rbx
  unsigned int v9; // eax
  unsigned int *v10; // r12
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 *v14; // rax
  unsigned __int64 v15; // rax
  unsigned int v16; // ecx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r15
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 v26; // r13
  int v27; // eax
  __int64 v28; // r13
  __int64 v29; // rsi
  __int64 *v30; // r12
  __int64 *i; // r15
  __int64 v32; // r14
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // r12
  int v36; // r15d
  __int64 *v37; // rax
  unsigned int v38; // esi
  __int64 *v39; // rdx
  __int64 v40; // r10
  _QWORD *v41; // rax
  unsigned __int64 v42; // rcx
  unsigned __int64 v43; // rdx
  __int64 v44; // rdx
  unsigned __int64 v45; // rsi
  const __m128i *v46; // rax
  unsigned __int64 v47; // r8
  __m128i *v48; // rdx
  __int64 v49; // rdx
  const __m128i *v50; // r12
  const __m128i *v51; // rbx
  unsigned __int64 v52; // r13
  __m128i *result; // rax
  __int64 v54; // r15
  __int64 v55; // r13
  __int64 v56; // rsi
  int v57; // r13d
  unsigned __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // r14
  __int64 v61; // rdx
  unsigned __int64 v62; // rdi
  int v63; // r14d
  __int64 v64; // rax
  int v65; // eax
  __int64 v66; // rax
  __int64 v67; // r13
  __int64 v68; // rdx
  unsigned __int64 v69; // rdi
  int v70; // r13d
  __int64 v71; // rax
  int v72; // eax
  char *v73; // rbx
  int v74; // edx
  int v75; // edx
  int v76; // esi
  __int64 v77; // [rsp+0h] [rbp-C0h]
  unsigned int v78; // [rsp+Ch] [rbp-B4h]
  __int64 v79; // [rsp+18h] [rbp-A8h]
  __int64 v80; // [rsp+20h] [rbp-A0h]
  unsigned int *v81; // [rsp+28h] [rbp-98h]
  __int64 v82; // [rsp+28h] [rbp-98h]
  __int64 v83; // [rsp+28h] [rbp-98h]
  unsigned int v84; // [rsp+28h] [rbp-98h]
  __int64 v85; // [rsp+30h] [rbp-90h]
  __int64 v86; // [rsp+30h] [rbp-90h]
  __int64 v87; // [rsp+30h] [rbp-90h]
  __int64 v89; // [rsp+48h] [rbp-78h]
  __int64 **v90; // [rsp+48h] [rbp-78h]
  unsigned __int64 v91; // [rsp+58h] [rbp-68h] BYREF
  unsigned __int64 v92[3]; // [rsp+60h] [rbp-60h] BYREF
  char v93; // [rsp+78h] [rbp-48h]
  __int64 v94; // [rsp+80h] [rbp-40h]

  v6 = a1;
  v7 = *(_QWORD *)(a2 + 128);
  v8 = v7 + 32LL * *(unsigned int *)(a2 + 136);
  v9 = *(_DWORD *)(a1 + 64);
  if ( v7 != v8 )
  {
    v10 = (unsigned int *)(a1 + 56);
    v89 = a1 + 72;
    do
    {
      if ( *(_DWORD *)(v6 + 68) <= v9 )
      {
        v66 = sub_C8D7D0((__int64)v10, v89, 0, 0x20u, v92, a6);
        v67 = v66 + 32LL * *(unsigned int *)(v6 + 64);
        if ( v67 )
        {
          *(_DWORD *)v67 = *(_DWORD *)v7;
          *(_QWORD *)(v67 + 8) = *(_QWORD *)(v7 + 8);
          v68 = *(_QWORD *)(v7 + 16);
          *(_QWORD *)(v67 + 16) = v68;
          if ( v68 )
          {
            v86 = v66;
            sub_2D23AB0((__int64 *)(v67 + 16));
            v66 = v86;
          }
          *(_QWORD *)(v67 + 24) = *(_QWORD *)(v7 + 24);
        }
        v87 = v66;
        sub_2D296B0(v10, v66);
        v69 = *(_QWORD *)(v6 + 56);
        v70 = v92[0];
        v71 = v87;
        if ( v89 != v69 )
        {
          _libc_free(v69);
          v71 = v87;
        }
        *(_QWORD *)(v6 + 56) = v71;
        v72 = *(_DWORD *)(v6 + 64);
        *(_DWORD *)(v6 + 68) = v70;
        v9 = v72 + 1;
        *(_DWORD *)(v6 + 64) = v9;
      }
      else
      {
        v11 = *(_QWORD *)(v6 + 56) + 32LL * v9;
        if ( v11 )
        {
          *(_DWORD *)v11 = *(_DWORD *)v7;
          *(_QWORD *)(v11 + 8) = *(_QWORD *)(v7 + 8);
          v12 = *(_QWORD *)(v7 + 16);
          *(_QWORD *)(v11 + 16) = v12;
          if ( v12 )
            sub_B96E90(v11 + 16, v12, 1);
          *(_QWORD *)(v11 + 24) = *(_QWORD *)(v7 + 24);
          v9 = *(_DWORD *)(v6 + 64);
        }
        *(_DWORD *)(v6 + 64) = ++v9;
      }
      v7 += 32;
    }
    while ( v8 != v7 );
  }
  *(_DWORD *)(v6 + 104) = v9;
  v13 = v6;
  v77 = v6 + 112;
  v85 = v6 + 72;
  v90 = *(__int64 ***)(a2 + 88);
  if ( v90 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = v90[1];
        if ( ((unsigned __int8)v14 & 4) == 0 )
        {
          v15 = (unsigned __int64)v14 & 0xFFFFFFFFFFFFFFF8LL;
          v16 = *(_DWORD *)(v13 + 64);
          v17 = *(_QWORD *)(v15 + 64);
          v91 = v15;
          v78 = v16;
          if ( v17 )
          {
            v19 = sub_B14240(v17);
            v20 = v18;
            if ( v18 != v19 )
            {
              while ( *(_BYTE *)(v19 + 32) )
              {
                v19 = *(_QWORD *)(v19 + 8);
                if ( v18 == v19 )
                  goto LABEL_35;
              }
              if ( v18 != v19 )
              {
                v81 = (unsigned int *)(v13 + 56);
                while ( 1 )
                {
                  v92[0] = v19 | 4;
                  v21 = sub_2D2B810((_QWORD *)(a2 + 72), (v19 | 4uLL) % *(_QWORD *)(a2 + 80), v92, v19 | 4);
                  if ( v21 )
                  {
                    v22 = *v21;
                    if ( v22 )
                    {
                      v23 = *(_QWORD *)(v22 + 16);
                      v24 = v23 + 32LL * *(unsigned int *)(v22 + 24);
                      if ( v24 != v23 )
                        break;
                    }
                  }
                  do
                  {
                    v19 = *(_QWORD *)(v19 + 8);
                    if ( v20 == v19 )
                      goto LABEL_35;
LABEL_33:
                    ;
                  }
                  while ( *(_BYTE *)(v19 + 32) );
                  if ( v20 == v19 )
                    goto LABEL_35;
                }
                v79 = v20;
                v80 = v19;
                v25 = v13;
                do
                {
                  v26 = *(unsigned int *)(v25 + 64);
                  v27 = v26;
                  if ( *(_DWORD *)(v25 + 68) <= (unsigned int)v26 )
                  {
                    v54 = sub_C8D7D0((__int64)v81, v85, 0, 0x20u, v92, a6);
                    v55 = v54 + 32LL * *(unsigned int *)(v25 + 64);
                    if ( v55 )
                    {
                      *(_DWORD *)v55 = *(_DWORD *)v23;
                      *(_QWORD *)(v55 + 8) = *(_QWORD *)(v23 + 8);
                      v56 = *(_QWORD *)(v23 + 16);
                      *(_QWORD *)(v55 + 16) = v56;
                      if ( v56 )
                        sub_B96E90(v55 + 16, v56, 1);
                      *(_QWORD *)(v55 + 24) = *(_QWORD *)(v23 + 24);
                    }
                    sub_2D296B0(v81, v54);
                    v57 = v92[0];
                    v58 = *(_QWORD *)(v25 + 56);
                    if ( v85 != v58 )
                      _libc_free(v58);
                    ++*(_DWORD *)(v25 + 64);
                    *(_QWORD *)(v25 + 56) = v54;
                    *(_DWORD *)(v25 + 68) = v57;
                  }
                  else
                  {
                    v28 = *(_QWORD *)(v25 + 56) + 32 * v26;
                    if ( v28 )
                    {
                      *(_DWORD *)v28 = *(_DWORD *)v23;
                      *(_QWORD *)(v28 + 8) = *(_QWORD *)(v23 + 8);
                      v29 = *(_QWORD *)(v23 + 16);
                      *(_QWORD *)(v28 + 16) = v29;
                      if ( v29 )
                        sub_B96E90(v28 + 16, v29, 1);
                      *(_QWORD *)(v28 + 24) = *(_QWORD *)(v23 + 24);
                      v27 = *(_DWORD *)(v25 + 64);
                    }
                    *(_DWORD *)(v25 + 64) = v27 + 1;
                  }
                  v23 += 32;
                }
                while ( v24 != v23 );
                v13 = v25;
                v20 = v79;
                v19 = *(_QWORD *)(v80 + 8);
                if ( v79 == v19 )
                  goto LABEL_35;
                goto LABEL_33;
              }
            }
LABEL_35:
            v16 = *(_DWORD *)(v13 + 64);
          }
          v30 = v90[2];
          for ( i = &v30[4 * *((unsigned int *)v90 + 6)]; i != v30; v30 += 4 )
          {
            if ( *(_DWORD *)(v13 + 68) <= v16 )
            {
              v59 = sub_C8D7D0(v13 + 56, v85, 0, 0x20u, v92, a6);
              v60 = v59 + 32LL * *(unsigned int *)(v13 + 64);
              if ( v60 )
              {
                *(_DWORD *)v60 = *(_DWORD *)v30;
                *(_QWORD *)(v60 + 8) = v30[1];
                v61 = v30[2];
                *(_QWORD *)(v60 + 16) = v61;
                if ( v61 )
                {
                  v82 = v59;
                  sub_2D23AB0((__int64 *)(v60 + 16));
                  v59 = v82;
                }
                *(_QWORD *)(v60 + 24) = v30[3];
              }
              v83 = v59;
              sub_2D296B0((unsigned int *)(v13 + 56), v59);
              v62 = *(_QWORD *)(v13 + 56);
              v63 = v92[0];
              v64 = v83;
              if ( v85 != v62 )
              {
                _libc_free(v62);
                v64 = v83;
              }
              *(_QWORD *)(v13 + 56) = v64;
              v65 = *(_DWORD *)(v13 + 64);
              *(_DWORD *)(v13 + 68) = v63;
              v16 = v65 + 1;
              *(_DWORD *)(v13 + 64) = v65 + 1;
            }
            else
            {
              v32 = *(_QWORD *)(v13 + 56) + 32LL * v16;
              if ( v32 )
              {
                *(_DWORD *)v32 = *(_DWORD *)v30;
                *(_QWORD *)(v32 + 8) = v30[1];
                v33 = v30[2];
                *(_QWORD *)(v32 + 16) = v33;
                if ( v33 )
                  sub_B96E90(v32 + 16, v33, 1);
                *(_QWORD *)(v32 + 24) = v30[3];
                v16 = *(_DWORD *)(v13 + 64);
              }
              *(_DWORD *)(v13 + 64) = ++v16;
            }
          }
          if ( v78 != v16 )
            break;
        }
        v90 = (__int64 **)*v90;
        if ( !v90 )
          goto LABEL_49;
      }
      a5 = *(unsigned int *)(v13 + 136);
      if ( !(_DWORD)a5 )
        break;
      v34 = v91;
      v35 = *(_QWORD *)(v13 + 120);
      v36 = 1;
      v37 = 0;
      v38 = (a5 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
      v39 = (__int64 *)(v35 + 16LL * v38);
      v40 = *v39;
      if ( v91 != *v39 )
      {
        while ( v40 != -4096 )
        {
          if ( v40 == -8192 && !v37 )
            v37 = v39;
          a6 = (unsigned int)(v36 + 1);
          v38 = (a5 - 1) & (v36 + v38);
          v39 = (__int64 *)(v35 + 16LL * v38);
          v40 = *v39;
          if ( v91 == *v39 )
            goto LABEL_47;
          ++v36;
        }
        if ( !v37 )
          v37 = v39;
        v74 = *(_DWORD *)(v13 + 128);
        ++*(_QWORD *)(v13 + 112);
        v75 = v74 + 1;
        v92[0] = (unsigned __int64)v37;
        if ( 4 * v75 < (unsigned int)(3 * a5) )
        {
          if ( (int)a5 - *(_DWORD *)(v13 + 132) - v75 > (unsigned int)a5 >> 3 )
          {
LABEL_99:
            *(_DWORD *)(v13 + 128) = v75;
            if ( *v37 != -4096 )
              --*(_DWORD *)(v13 + 132);
            *v37 = v34;
            v41 = v37 + 1;
            *v41 = 0;
            goto LABEL_48;
          }
          v84 = v16;
          v76 = a5;
LABEL_104:
          sub_2D2C730(v77, v76);
          sub_2D28930(v77, (__int64 *)&v91, v92);
          v34 = v91;
          v16 = v84;
          v75 = *(_DWORD *)(v13 + 128) + 1;
          v37 = (__int64 *)v92[0];
          goto LABEL_99;
        }
LABEL_103:
        v84 = v16;
        v76 = 2 * a5;
        goto LABEL_104;
      }
LABEL_47:
      v41 = v39 + 1;
LABEL_48:
      *((_DWORD *)v41 + 1) = v16;
      *(_DWORD *)v41 = v78;
      v90 = (__int64 **)*v90;
      if ( !v90 )
      {
LABEL_49:
        v6 = v13;
        goto LABEL_50;
      }
    }
    ++*(_QWORD *)(v13 + 112);
    v92[0] = 0;
    goto LABEL_103;
  }
LABEL_50:
  v42 = *(unsigned int *)(v6 + 12);
  v43 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 48)) >> 3) + 1;
  if ( v43 > v42 )
  {
    sub_C8D5F0(v6, (const void *)(v6 + 16), v43, 0x28u, a5, a6);
    v42 = *(unsigned int *)(v6 + 12);
  }
  v44 = *(unsigned int *)(v6 + 8);
  v45 = *(_QWORD *)v6;
  v93 = 0;
  v92[0] = 0;
  v46 = (const __m128i *)v92;
  v47 = v44 + 1;
  v94 = 0;
  if ( v44 + 1 > v42 )
  {
    if ( v45 > (unsigned __int64)v92 || (unsigned __int64)v92 >= v45 + 40 * v44 )
    {
      sub_C8D5F0(v6, (const void *)(v6 + 16), v47, 0x28u, v47, a6);
      v45 = *(_QWORD *)v6;
      v44 = *(unsigned int *)(v6 + 8);
      v46 = (const __m128i *)v92;
    }
    else
    {
      v73 = (char *)v92 - v45;
      sub_C8D5F0(v6, (const void *)(v6 + 16), v47, 0x28u, v47, a6);
      v45 = *(_QWORD *)v6;
      v44 = *(unsigned int *)(v6 + 8);
      v46 = (const __m128i *)&v73[*(_QWORD *)v6];
    }
  }
  v48 = (__m128i *)(v45 + 40 * v44);
  *v48 = _mm_loadu_si128(v46);
  v48[1] = _mm_loadu_si128(v46 + 1);
  v48[2].m128i_i64[0] = v46[2].m128i_i64[0];
  v49 = (unsigned int)(*(_DWORD *)(v6 + 8) + 1);
  *(_DWORD *)(v6 + 8) = v49;
  v50 = *(const __m128i **)(a2 + 56);
  v51 = *(const __m128i **)(a2 + 48);
  v52 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v50 - (char *)v51) >> 3);
  if ( v52 + v49 > *(unsigned int *)(v6 + 12) )
  {
    sub_C8D5F0(v6, (const void *)(v6 + 16), v52 + v49, 0x28u, v52 + v49, a6);
    v49 = *(unsigned int *)(v6 + 8);
  }
  result = (__m128i *)(*(_QWORD *)v6 + 40 * v49);
  if ( v51 != v50 )
  {
    do
    {
      if ( result )
      {
        *result = _mm_loadu_si128(v51);
        result[1] = _mm_loadu_si128(v51 + 1);
        result[2].m128i_i64[0] = v51[2].m128i_i64[0];
      }
      v51 = (const __m128i *)((char *)v51 + 40);
      result = (__m128i *)((char *)result + 40);
    }
    while ( v50 != v51 );
    LODWORD(v49) = *(_DWORD *)(v6 + 8);
  }
  *(_DWORD *)(v6 + 8) = v52 + v49;
  return result;
}
