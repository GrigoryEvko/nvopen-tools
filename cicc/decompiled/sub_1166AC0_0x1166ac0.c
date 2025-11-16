// Function: sub_1166AC0
// Address: 0x1166ac0
//
unsigned __int8 *__fastcall sub_1166AC0(__m128i *a1, __int64 a2)
{
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __m128i v6; // xmm3
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r9
  __int64 v17; // rbx
  unsigned int v18; // eax
  unsigned int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // r9
  _BYTE *v22; // rax
  _BYTE *v23; // r8
  __int64 v24; // rdx
  bool v25; // zf
  unsigned int v26; // eax
  __int64 v27; // rcx
  __m128i v28; // xmm5
  __int64 v29; // rax
  unsigned __int64 v30; // xmm6_8
  __m128i v31; // xmm7
  unsigned int v32; // edx
  unsigned int v33; // esi
  __int64 v34; // rax
  bool v35; // al
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  _QWORD *v39; // rsi
  int v40; // eax
  unsigned int v41; // eax
  __int64 v42; // rdx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rax
  __int32 v46; // eax
  __int64 v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rcx
  __int64 v52; // rcx
  __int64 v53; // r12
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r14
  __int64 v60; // rbx
  __int64 v61; // r15
  __int64 v62; // r13
  __m128i v63; // xmm5
  __int64 v64; // rax
  unsigned __int64 v65; // xmm6_8
  __m128i v66; // xmm7
  __m128i v67; // rax
  __int64 v68; // rbx
  __int64 v69; // r12
  __int64 v70; // rdx
  unsigned int v71; // esi
  __m128i *v72; // rax
  __m128i *v73; // rdx
  __m128i *i; // rdx
  __int64 v75; // r12
  __int64 v76; // rax
  __int64 v77; // rdi
  unsigned int v78; // edx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rcx
  __int64 v83; // rax
  __int64 v84; // rdi
  bool v85; // [rsp+0h] [rbp-150h]
  __int64 v86; // [rsp+0h] [rbp-150h]
  unsigned __int32 v87; // [rsp+8h] [rbp-148h]
  __int64 v88; // [rsp+8h] [rbp-148h]
  unsigned int v89; // [rsp+8h] [rbp-148h]
  bool v90; // [rsp+10h] [rbp-140h]
  unsigned int v91; // [rsp+10h] [rbp-140h]
  __int64 v92; // [rsp+18h] [rbp-138h]
  unsigned int v93; // [rsp+18h] [rbp-138h]
  __int64 v94; // [rsp+20h] [rbp-130h] BYREF
  __int64 v95; // [rsp+28h] [rbp-128h] BYREF
  __int64 v96[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v97; // [rsp+50h] [rbp-100h]
  __int64 v98; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v99; // [rsp+68h] [rbp-E8h]
  __int16 v100; // [rsp+80h] [rbp-D0h]
  __m128i v101; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v102; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v103; // [rsp+B0h] [rbp-A0h]
  __m128i v104; // [rsp+C0h] [rbp-90h]
  __int64 v105; // [rsp+D0h] [rbp-80h]

  v4 = _mm_loadu_si128(a1 + 6);
  v5 = _mm_loadu_si128(a1 + 7);
  v6 = _mm_loadu_si128(a1 + 9);
  v7 = a1[10].m128i_i64[0];
  v103 = _mm_loadu_si128(a1 + 8);
  v8 = *(_QWORD *)(a2 - 64);
  v103.m128i_i64[1] = a2;
  v9 = *(unsigned __int8 **)(a2 - 32);
  v105 = v7;
  v101 = v4;
  v102 = v5;
  v104 = v6;
  v10 = sub_101AFC0(v8, v9, &v101);
  if ( v10 )
    return sub_F162A0((__int64)a1, a2, v10);
  v12 = (__int64)sub_F0F270((__int64)a1, (unsigned __int8 *)a2);
  if ( !v12 )
  {
    v12 = (__int64)sub_1166190(a1, a2);
    if ( !v12 )
    {
      v12 = (__int64)sub_11597C0((__int64)a1, a2, v13, v14, v15, v16);
      if ( !v12 )
      {
        v17 = *(_QWORD *)(a2 - 32);
        v92 = *(_QWORD *)(a2 - 64);
        if ( *(_BYTE *)v17 == 17 )
        {
          v18 = *(_DWORD *)(v17 + 32);
          v19 = v18 - 1;
          v20 = *(_QWORD *)(v17 + 24);
          if ( v18 > 0x40 )
            v20 = *(_QWORD *)(v20 + 8LL * (v19 >> 6));
          v21 = v17 + 24;
          if ( (v20 & (1LL << ((unsigned __int8)v18 - 1))) != 0 )
            goto LABEL_37;
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v17 + 8) + 8LL) - 17 > 1 )
            goto LABEL_17;
        }
        else
        {
          v20 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v17 + 8) + 8LL) - 17;
          if ( (unsigned int)v20 > 1 || *(_BYTE *)v17 > 0x15u )
            goto LABEL_17;
        }
        v22 = sub_AD7630(v17, 1, v20);
        v23 = v22;
        if ( !v22 || *v22 != 17 )
          goto LABEL_17;
        v18 = *((_DWORD *)v22 + 8);
        v19 = v18 - 1;
        v24 = *((_QWORD *)v23 + 3);
        if ( v18 > 0x40 )
          v24 = *(_QWORD *)(v24 + 8LL * (v19 >> 6));
        if ( (v24 & (1LL << ((unsigned __int8)v18 - 1))) == 0 )
        {
LABEL_17:
          v25 = *(_BYTE *)a2 == 52;
          v101.m128i_i64[0] = 0;
          v101.m128i_i64[1] = (__int64)&v94;
          v102.m128i_i64[0] = (__int64)&v95;
          if ( v25 )
          {
            v56 = *(_QWORD *)(a2 - 64);
            v57 = *(_QWORD *)(v56 + 16);
            if ( v57 )
            {
              if ( !*(_QWORD *)(v57 + 8) )
              {
                if ( (unsigned __int8)sub_10E40A0(&v101, (unsigned __int8 *)v56) )
                {
                  v58 = *(_QWORD *)(a2 - 32);
                  if ( v58 )
                  {
                    *(_QWORD *)v102.m128i_i64[0] = v58;
                    v59 = a1[2].m128i_i64[0];
                    v100 = 257;
                    v60 = v95;
                    v97 = 257;
                    v61 = v94;
                    v62 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v59 + 80) + 16LL))(
                            *(_QWORD *)(v59 + 80),
                            23,
                            v94,
                            v95);
                    if ( !v62 )
                    {
                      v103.m128i_i16[0] = 257;
                      v62 = sub_B504D0(23, v61, v60, (__int64)&v101, 0, 0);
                      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v59 + 88) + 16LL))(
                        *(_QWORD *)(v59 + 88),
                        v62,
                        v96,
                        *(_QWORD *)(v59 + 56),
                        *(_QWORD *)(v59 + 64));
                      v68 = *(_QWORD *)v59;
                      v69 = *(_QWORD *)v59 + 16LL * *(unsigned int *)(v59 + 8);
                      if ( *(_QWORD *)v59 != v69 )
                      {
                        do
                        {
                          v70 = *(_QWORD *)(v68 + 8);
                          v71 = *(_DWORD *)v68;
                          v68 += 16;
                          sub_B99FD0(v62, v71, v70);
                        }
                        while ( v69 != v68 );
                      }
                    }
                    return sub_B505E0(v62, (__int64)&v98, 0, 0);
                  }
                }
              }
            }
          }
          v26 = sub_BCB060(*(_QWORD *)(a2 + 8));
          v99 = v26;
          v27 = 1LL << ((unsigned __int8)v26 - 1);
          if ( v26 > 0x40 )
          {
            v86 = 1LL << ((unsigned __int8)v26 - 1);
            v89 = v26 - 1;
            sub_C43690((__int64)&v98, 0, 0);
            v27 = v86;
            if ( v99 > 0x40 )
            {
              *(_QWORD *)(v98 + 8LL * (v89 >> 6)) |= v86;
              goto LABEL_21;
            }
          }
          else
          {
            v98 = 0;
          }
          v98 |= v27;
LABEL_21:
          v28 = _mm_loadu_si128(a1 + 7);
          v29 = a1[10].m128i_i64[0];
          v30 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
          v101 = _mm_loadu_si128(a1 + 6);
          v31 = _mm_loadu_si128(a1 + 9);
          v105 = v29;
          v103.m128i_i64[0] = v30;
          v102 = v28;
          v103.m128i_i64[1] = a2;
          v104 = v31;
          if ( (unsigned __int8)sub_9AC230(v17, (__int64)&v98, &v101, 0) )
          {
            v63 = _mm_loadu_si128(a1 + 7);
            v64 = a1[10].m128i_i64[0];
            v65 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
            v101 = _mm_loadu_si128(a1 + 6);
            v66 = _mm_loadu_si128(a1 + 9);
            v105 = v64;
            v103.m128i_i64[0] = v65;
            v102 = v63;
            v103.m128i_i64[1] = a2;
            v104 = v66;
            if ( (unsigned __int8)sub_9AC230(v92, (__int64)&v98, &v101, 0) )
            {
              v67.m128i_i64[0] = (__int64)sub_BD5D20(a2);
              v103.m128i_i16[0] = 261;
              v101 = v67;
              v12 = sub_B504D0(22, v92, v17, (__int64)&v101, 0, 0);
              goto LABEL_33;
            }
          }
          v85 = *(_BYTE *)v17 == 16 || *(_BYTE *)v17 == 11;
          if ( v85 )
          {
            v87 = *(_DWORD *)(*(_QWORD *)(v17 + 8) + 32LL);
            if ( v87 )
            {
              v90 = 0;
              v32 = 0;
              do
              {
                v93 = v32;
                v36 = sub_AD69F0((unsigned __int8 *)v17, v32);
                if ( !v36 )
                  goto LABEL_33;
                if ( *(_BYTE *)v36 == 17 )
                {
                  v33 = *(_DWORD *)(v36 + 32);
                  v34 = *(_QWORD *)(v36 + 24);
                  if ( v33 > 0x40 )
                    v34 = *(_QWORD *)(v34 + 8LL * ((v33 - 1) >> 6));
                  v25 = (v34 & (1LL << ((unsigned __int8)v33 - 1))) == 0;
                  v35 = v90;
                  if ( !v25 )
                    v35 = v85;
                  v90 = v35;
                }
                v32 = v93 + 1;
              }
              while ( v93 + 1 != v87 );
              if ( !v90 )
                goto LABEL_33;
              v101.m128i_i64[1] = 0x1000000000LL;
              v72 = &v102;
              v73 = &v102;
              v101.m128i_i64[0] = (__int64)&v102;
              if ( v87 > 0x10 )
              {
                sub_C8D5F0((__int64)&v101, &v102, v87, 8u, v37, v38);
                v73 = (__m128i *)v101.m128i_i64[0];
                v72 = (__m128i *)(v101.m128i_i64[0] + 8LL * v101.m128i_u32[2]);
              }
              for ( i = (__m128i *)((char *)v73 + 8 * v87); i != v72; v72 = (__m128i *)((char *)v72 + 8) )
              {
                if ( v72 )
                  v72->m128i_i64[0] = 0;
              }
              v75 = 0;
              v101.m128i_i32[2] = v87;
              do
              {
                v76 = sub_AD69F0((unsigned __int8 *)v17, (unsigned int)v75);
                *(_QWORD *)(v101.m128i_i64[0] + 8 * v75) = v76;
                v77 = *(_QWORD *)(v101.m128i_i64[0] + 8 * v75);
                if ( *(_BYTE *)v77 == 17 )
                {
                  v78 = *(_DWORD *)(v77 + 32);
                  v79 = *(_QWORD *)(v77 + 24);
                  if ( v78 > 0x40 )
                    v79 = *(_QWORD *)(v79 + 8LL * ((v78 - 1) >> 6));
                  if ( (v79 & (1LL << ((unsigned __int8)v78 - 1))) != 0 )
                  {
                    v80 = sub_AD6890(v77, 0);
                    *(_QWORD *)(v101.m128i_i64[0] + 8 * v75) = v80;
                  }
                }
                ++v75;
              }
              while ( v87 != v75 );
              v81 = v101.m128i_u32[2];
              v82 = sub_AD3730((__int64 *)v101.m128i_i64[0], v101.m128i_u32[2]);
              if ( v17 != v82 )
              {
                v81 = a2;
                v83 = sub_F20660((__int64)a1, a2, 1u, v82);
                v84 = v101.m128i_i64[0];
                v12 = v83;
                if ( (__m128i *)v101.m128i_i64[0] == &v102 )
                  goto LABEL_33;
                goto LABEL_100;
              }
              v84 = v101.m128i_i64[0];
              if ( (__m128i *)v101.m128i_i64[0] != &v102 )
LABEL_100:
                _libc_free(v84, v81);
            }
          }
LABEL_33:
          if ( v99 > 0x40 )
          {
            if ( v98 )
              j_j___libc_free_0_0(v98);
          }
          return (unsigned __int8 *)v12;
        }
        v21 = (__int64)(v23 + 24);
LABEL_37:
        v39 = *(_QWORD **)v21;
        if ( v18 <= 0x40 )
        {
          if ( v39 == (_QWORD *)(1LL << v19) )
            goto LABEL_17;
        }
        else
        {
          v91 = v19;
          if ( (v39[v19 >> 6] & (1LL << v19)) != 0 )
          {
            v88 = v21;
            v40 = sub_C44590(v21);
            v21 = v88;
            if ( v40 == v91 )
              goto LABEL_17;
          }
        }
        v41 = *(_DWORD *)(v21 + 8);
        v99 = v41;
        if ( v41 > 0x40 )
        {
          sub_C43780((__int64)&v98, (const void **)v21);
          v41 = v99;
          if ( v99 > 0x40 )
          {
            sub_C43D10((__int64)&v98);
LABEL_45:
            sub_C46250((__int64)&v98);
            v46 = v99;
            v47 = *(_QWORD *)(a2 + 8);
            v99 = 0;
            v101.m128i_i32[2] = v46;
            v101.m128i_i64[0] = v98;
            v48 = sub_AD8D80(v47, (__int64)&v101);
            if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
              v49 = *(_QWORD *)(a2 - 8);
            else
              v49 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
            v50 = *(_QWORD *)(v49 + 32);
            if ( v50 )
            {
              v51 = *(_QWORD *)(v49 + 40);
              **(_QWORD **)(v49 + 48) = v51;
              if ( v51 )
                *(_QWORD *)(v51 + 16) = *(_QWORD *)(v49 + 48);
            }
            *(_QWORD *)(v49 + 32) = v48;
            if ( v48 )
            {
              v52 = *(_QWORD *)(v48 + 16);
              *(_QWORD *)(v49 + 40) = v52;
              if ( v52 )
                *(_QWORD *)(v52 + 16) = v49 + 40;
              *(_QWORD *)(v49 + 48) = v48 + 16;
              *(_QWORD *)(v48 + 16) = v49 + 32;
            }
            if ( *(_BYTE *)v50 > 0x1Cu )
            {
              v53 = a1[2].m128i_i64[1];
              v96[0] = v50;
              v54 = v53 + 2096;
              sub_11604F0(v54, v96);
              v55 = *(_QWORD *)(v50 + 16);
              if ( v55 )
              {
                if ( !*(_QWORD *)(v55 + 8) )
                {
                  v96[0] = *(_QWORD *)(v55 + 24);
                  sub_11604F0(v54, v96);
                }
              }
            }
            if ( v101.m128i_i32[2] > 0x40u && v101.m128i_i64[0] )
              j_j___libc_free_0_0(v101.m128i_i64[0]);
            if ( v99 > 0x40 && v98 )
              j_j___libc_free_0_0(v98);
            return (unsigned __int8 *)a2;
          }
          v42 = v98;
        }
        else
        {
          v42 = *(_QWORD *)v21;
        }
        v43 = ~v42;
        v44 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v41;
        v25 = v41 == 0;
        v45 = 0;
        if ( !v25 )
          v45 = v44;
        v98 = v45 & v43;
        goto LABEL_45;
      }
    }
  }
  return (unsigned __int8 *)v12;
}
