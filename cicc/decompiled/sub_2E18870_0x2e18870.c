// Function: sub_2E18870
// Address: 0x2e18870
//
void __fastcall sub_2E18870(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rcx
  unsigned int v10; // edi
  unsigned int v11; // esi
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rsi
  char v22; // al
  __int64 v23; // r13
  __int64 v24; // rcx
  char *v25; // rax
  const __m128i *v26; // rsi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  __m128i *v30; // rcx
  const __m128i *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // r15
  __int64 v34; // r13
  unsigned __int64 v35; // r14
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // rax
  unsigned __int64 v43; // r15
  __int64 v44; // r14
  __int64 *v45; // rax
  __int64 v46; // rcx
  __int64 *v47; // rdx
  _QWORD *v48; // rdi
  __int64 v49; // r13
  __int64 *v50; // rax
  __int64 v51; // rsi
  char v52; // dl
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // rax
  _QWORD *v59; // r8
  unsigned int v60; // ecx
  __int64 v61; // rsi
  __int64 *v62; // r9
  __int64 *v63; // [rsp+8h] [rbp-148h]
  __int64 v64; // [rsp+10h] [rbp-140h]
  unsigned __int64 v65; // [rsp+20h] [rbp-130h]
  __int64 *v66; // [rsp+20h] [rbp-130h]
  __int64 v67; // [rsp+20h] [rbp-130h]
  __int64 v68; // [rsp+28h] [rbp-128h]
  unsigned __int64 v69; // [rsp+28h] [rbp-128h]
  unsigned __int64 v70; // [rsp+30h] [rbp-120h]
  __int64 v71; // [rsp+30h] [rbp-120h]
  __m128i v73; // [rsp+40h] [rbp-110h] BYREF
  unsigned __int64 v74; // [rsp+50h] [rbp-100h]
  unsigned __int64 v75; // [rsp+58h] [rbp-F8h]
  __m128i v76; // [rsp+60h] [rbp-F0h] BYREF
  const __m128i *v77; // [rsp+70h] [rbp-E0h]
  __int64 v78; // [rsp+78h] [rbp-D8h]
  __int64 v79; // [rsp+80h] [rbp-D0h] BYREF
  char *v80; // [rsp+88h] [rbp-C8h]
  __int64 v81; // [rsp+90h] [rbp-C0h]
  int v82; // [rsp+98h] [rbp-B8h]
  char v83; // [rsp+9Ch] [rbp-B4h]
  char v84; // [rsp+A0h] [rbp-B0h] BYREF

  v70 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = (__int64 *)sub_2E09D00((__int64 *)a2, a3 & 0xFFFFFFFFFFFFFFF8LL);
  v8 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( v7 != (__int64 *)v8 )
  {
    v9 = v70;
    v10 = *(_DWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    v11 = *(_DWORD *)(v70 + 24);
    if ( (unsigned __int64)(v10 | (*v7 >> 1) & 3) <= v11 && v70 == (v7[1] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( (__int64 *)v8 == v7 + 3 )
        return;
      v10 = *(_DWORD *)((v7[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v7 += 3;
    }
    if ( v10 <= v11 )
    {
      v12 = v7[1];
      v71 = v7[2];
      if ( v71 )
      {
        v13 = *(_QWORD *)(a1 + 32);
        v14 = *(_QWORD *)(v9 + 16);
        if ( v14 )
        {
          v15 = *(_QWORD *)(v14 + 24);
        }
        else
        {
          v58 = *(unsigned int *)(v13 + 304);
          v59 = *(_QWORD **)(v13 + 296);
          if ( *(_DWORD *)(v13 + 304) )
          {
            v60 = v11 | (a3 >> 1) & 3;
            do
            {
              while ( 1 )
              {
                v61 = v58 >> 1;
                v62 = &v59[2 * (v58 >> 1)];
                if ( v60 < (*(_DWORD *)((*v62 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v62 >> 1) & 3) )
                  break;
                v59 = v62 + 2;
                v58 = v58 - v61 - 1;
                if ( v58 <= 0 )
                  goto LABEL_92;
              }
              v58 >>= 1;
            }
            while ( v61 > 0 );
          }
LABEL_92:
          v15 = *(v59 - 1);
        }
        if ( (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12 >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)(v13 + 152) + 16LL * *(unsigned int *)(v15 + 24) + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*(__int64 *)(*(_QWORD *)(v13 + 152) + 16LL * *(unsigned int *)(v15 + 24) + 8) >> 1)
                                                                                              & 3) )
        {
          sub_2E0C3B0(a2, a3, v12, 0);
          if ( a4 )
          {
            v57 = *(unsigned int *)(a4 + 8);
            if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
            {
              sub_C8D5F0(a4, (const void *)(a4 + 16), v57 + 1, 8u, v55, v56);
              v57 = *(unsigned int *)(a4 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v57) = v12;
            ++*(_DWORD *)(a4 + 8);
          }
        }
        else
        {
          v65 = v15;
          v68 = *(_QWORD *)(*(_QWORD *)(v13 + 152) + 16LL * *(unsigned int *)(v15 + 24) + 8);
          sub_2E0C3B0(a2, a3, v68, 0);
          v17 = v65;
          if ( a4 )
          {
            v18 = *(unsigned int *)(a4 + 8);
            v16 = v68;
            if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
            {
              v67 = v68;
              v69 = v17;
              sub_C8D5F0(a4, (const void *)(a4 + 16), v18 + 1, 8u, v17, v16);
              v18 = *(unsigned int *)(a4 + 8);
              v16 = v67;
              v17 = v69;
            }
            *(_QWORD *)(*(_QWORD *)a4 + 8 * v18) = v16;
            ++*(_DWORD *)(a4 + 8);
          }
          v19 = *(unsigned int *)(v17 + 120);
          v79 = 0;
          v80 = &v84;
          v20 = *(_QWORD *)(v17 + 112) + 8 * v19;
          v21 = *(__int64 **)(v17 + 112);
          v66 = v21;
          v22 = 1;
          v81 = 16;
          v82 = 0;
          v83 = 1;
          v63 = (__int64 *)v20;
          if ( v21 != (__int64 *)v20 )
          {
            while ( 1 )
            {
              v23 = *v66;
              v24 = (__int64)&v79;
              v76.m128i_i64[1] = 0;
              v77 = 0;
              v76.m128i_i64[0] = (__int64)&v79;
              v78 = 0;
              if ( v22 )
              {
                v25 = v80;
                v24 = HIDWORD(v81);
                v19 = (__int64)&v80[8 * HIDWORD(v81)];
                if ( v80 != (char *)v19 )
                {
                  while ( v23 != *(_QWORD *)v25 )
                  {
                    v25 += 8;
                    if ( (char *)v19 == v25 )
                      goto LABEL_79;
                  }
                  goto LABEL_19;
                }
LABEL_79:
                if ( HIDWORD(v81) < (unsigned int)v81 )
                  break;
              }
              sub_C8CC70((__int64)&v79, v23, v19, v24, v17, v16);
              if ( (_BYTE)v19 )
                goto LABEL_77;
LABEL_19:
              v26 = v77;
              v27 = v76.m128i_u64[1];
              v74 = 0;
              v73 = (__m128i)v76.m128i_u64[0];
              v75 = 0;
              v28 = (unsigned __int64)v77 - v76.m128i_i64[1];
              if ( v77 == (const __m128i *)v76.m128i_i64[1] )
              {
                v19 = 0;
              }
              else
              {
                if ( v28 > 0x7FFFFFFFFFFFFFF8LL )
                  sub_4261EA(v76.m128i_i64[1], v77, v19);
                v29 = sub_22077B0((unsigned __int64)v77 - v76.m128i_i64[1]);
                v26 = v77;
                v27 = v76.m128i_u64[1];
                v19 = v29;
              }
              v73.m128i_i64[1] = v19;
              v74 = v19;
              v75 = v19 + v28;
              if ( (const __m128i *)v27 == v26 )
              {
                v17 = v19;
              }
              else
              {
                v30 = (__m128i *)v19;
                v31 = (const __m128i *)v27;
                do
                {
                  if ( v30 )
                  {
                    *v30 = _mm_loadu_si128(v31);
                    v30[1].m128i_i64[0] = v31[1].m128i_i64[0];
                  }
                  v31 = (const __m128i *)((char *)v31 + 24);
                  v30 = (__m128i *)((char *)v30 + 24);
                }
                while ( v31 != v26 );
                v17 = v19 + 8 * (((unsigned __int64)&v31[-2].m128i_u64[1] - v27) >> 3) + 24;
              }
              v74 = v17;
              if ( v27 )
              {
                j_j___libc_free_0(v27);
                v19 = v73.m128i_i64[1];
                v17 = v74;
              }
LABEL_30:
              while ( v17 != v19 )
              {
                v32 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 152LL)
                                + 16LL * *(unsigned int *)(*(_QWORD *)(v17 - 24) + 24LL));
                v33 = *v32;
                v34 = v32[1];
                v35 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
                v36 = (__int64 *)sub_2E09D00((__int64 *)a2, v35);
                v38 = 3LL * *(unsigned int *)(a2 + 8);
                v39 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
                if ( v36 == (__int64 *)v39 )
                  goto LABEL_65;
                v17 = *(unsigned int *)(v35 + 24);
                v38 = *(unsigned int *)((*v36 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                if ( (unsigned __int64)((unsigned int)v38 | (*v36 >> 1) & 3) > (unsigned int)v17 )
                {
                  v40 = 0;
                  v16 = 0;
                  goto LABEL_36;
                }
                v40 = v36[1];
                v16 = v36[2];
                if ( v35 != (v40 & 0xFFFFFFFFFFFFFFF8LL) )
                  goto LABEL_34;
                if ( (__int64 *)v39 != v36 + 3 )
                {
                  v38 = *(unsigned int *)((v36[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
                  v36 += 3;
LABEL_34:
                  v37 = 0;
                  if ( v35 == *(_QWORD *)(v16 + 8) )
                    v16 = 0;
LABEL_36:
                  if ( (unsigned int)v17 >= (unsigned int)v38 )
                    v40 = v36[1];
                }
                if ( v71 == v16 )
                {
                  if ( (*(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v40 >> 1) & 3) >= (*(_DWORD *)((v34 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v34 >> 1) & 3) )
                  {
                    sub_2E0C3B0(a2, v33, v34, 0);
                    if ( a4 )
                    {
                      v42 = *(unsigned int *)(a4 + 8);
                      if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
                      {
                        sub_C8D5F0(a4, (const void *)(a4 + 16), v42 + 1, 8u, v41, v16);
                        v42 = *(unsigned int *)(a4 + 8);
                      }
                      *(_QWORD *)(*(_QWORD *)a4 + 8 * v42) = v34;
                      ++*(_DWORD *)(a4 + 8);
                    }
                    v43 = v74;
                    while ( 1 )
                    {
                      v44 = *(_QWORD *)(v43 - 24);
                      if ( !*(_BYTE *)(v43 - 8) )
                      {
                        v45 = *(__int64 **)(v44 + 112);
                        *(_BYTE *)(v43 - 8) = 1;
                        *(_QWORD *)(v43 - 16) = v45;
                        goto LABEL_47;
                      }
                      while ( 1 )
                      {
                        v45 = *(__int64 **)(v43 - 16);
LABEL_47:
                        v46 = *(unsigned int *)(v44 + 120);
                        if ( v45 == (__int64 *)(*(_QWORD *)(v44 + 112) + 8 * v46) )
                          break;
                        v47 = v45 + 1;
                        *(_QWORD *)(v43 - 16) = v45 + 1;
                        v48 = (_QWORD *)v73.m128i_i64[0];
                        v49 = *v45;
                        if ( !*(_BYTE *)(v73.m128i_i64[0] + 28) )
                          goto LABEL_59;
                        v50 = *(__int64 **)(v73.m128i_i64[0] + 8);
                        v51 = *(unsigned int *)(v73.m128i_i64[0] + 20);
                        v47 = &v50[v51];
                        if ( v50 == v47 )
                        {
LABEL_61:
                          if ( (unsigned int)v51 < *(_DWORD *)(v73.m128i_i64[0] + 16) )
                          {
                            *(_DWORD *)(v73.m128i_i64[0] + 20) = v51 + 1;
                            *v47 = v49;
                            ++*v48;
LABEL_60:
                            v76.m128i_i64[0] = v49;
                            LOBYTE(v77) = 0;
                            sub_2E18730(&v73.m128i_u64[1], &v76);
                            v19 = v73.m128i_i64[1];
                            v17 = v74;
                            goto LABEL_30;
                          }
LABEL_59:
                          sub_C8CC70(v73.m128i_i64[0], v49, (__int64)v47, v46, v41, v16);
                          if ( v52 )
                            goto LABEL_60;
                        }
                        else
                        {
                          while ( v49 != *v50 )
                          {
                            if ( v47 == ++v50 )
                              goto LABEL_61;
                          }
                        }
                      }
                      v74 -= 24LL;
                      v19 = v73.m128i_i64[1];
                      v43 = v74;
                      if ( v74 == v73.m128i_i64[1] )
                        goto LABEL_64;
                    }
                  }
                  v38 = v33;
                  v64 = v40;
                  sub_2E0C3B0(a2, v33, v40, 0);
                  if ( a4 )
                  {
                    v53 = *(unsigned int *)(a4 + 8);
                    v54 = v64;
                    if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
                    {
                      v38 = a4 + 16;
                      sub_C8D5F0(a4, (const void *)(a4 + 16), v53 + 1, 8u, v53 + 1, v16);
                      v53 = *(unsigned int *)(a4 + 8);
                      v54 = v64;
                    }
                    v37 = *(_QWORD *)a4;
                    *(_QWORD *)(*(_QWORD *)a4 + 8 * v53) = v54;
                    ++*(_DWORD *)(a4 + 8);
                  }
                  v19 = v73.m128i_i64[1];
                  v74 -= 24LL;
                  v17 = v73.m128i_u64[1];
                  if ( v74 != v73.m128i_i64[1] )
                  {
LABEL_66:
                    sub_2E18770((unsigned __int64 *)&v73, v38, v19, v37, v17, v16);
                    v19 = v73.m128i_i64[1];
                    v17 = v74;
                  }
                }
                else
                {
LABEL_65:
                  v74 -= 24LL;
                  v19 = v73.m128i_i64[1];
                  if ( v74 != v73.m128i_i64[1] )
                    goto LABEL_66;
LABEL_64:
                  v17 = v19;
                }
              }
              if ( v17 )
                j_j___libc_free_0(v17);
              ++v66;
              v22 = v83;
              if ( v63 == v66 )
              {
                if ( !v83 )
                  _libc_free((unsigned __int64)v80);
                return;
              }
            }
            ++HIDWORD(v81);
            *(_QWORD *)v19 = v23;
            ++v79;
LABEL_77:
            v73.m128i_i64[0] = v23;
            LOBYTE(v74) = 0;
            sub_2E18730(&v76.m128i_u64[1], &v73);
            goto LABEL_19;
          }
        }
      }
    }
  }
}
