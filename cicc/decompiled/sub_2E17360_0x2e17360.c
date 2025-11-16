// Function: sub_2E17360
// Address: 0x2e17360
//
unsigned __int64 __fastcall sub_2E17360(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int128 a7)
{
  __m128i *v10; // rax
  __m128i *v11; // rsi
  __int64 v12; // r9
  __int64 v13; // r11
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 i; // rbx
  __int16 v19; // ax
  unsigned __int64 v20; // rcx
  unsigned __int64 j; // rax
  __int64 k; // rdi
  __int16 v23; // dx
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // r8
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // r13
  unsigned __int64 v32; // r14
  __int64 v33; // r8
  __int64 v34; // rax
  unsigned __int64 result; // rax
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rax
  char *v51; // rax
  int v52; // r13d
  __int128 v53; // [rsp-20h] [rbp-D0h]
  __int128 v54; // [rsp-20h] [rbp-D0h]
  unsigned int v55; // [rsp+Ch] [rbp-A4h]
  unsigned int v56; // [rsp+Ch] [rbp-A4h]
  unsigned int v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+10h] [rbp-A0h]
  unsigned int v59; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+10h] [rbp-A0h]
  unsigned int v61; // [rsp+10h] [rbp-A0h]
  unsigned int v62; // [rsp+18h] [rbp-98h]
  __int64 v63; // [rsp+18h] [rbp-98h]
  unsigned int v64; // [rsp+18h] [rbp-98h]
  __int64 v65; // [rsp+18h] [rbp-98h]
  __int64 v66; // [rsp+18h] [rbp-98h]
  _QWORD *v67; // [rsp+18h] [rbp-98h]
  __int64 v68; // [rsp+18h] [rbp-98h]
  __int64 v69; // [rsp+20h] [rbp-90h]
  _QWORD *v70; // [rsp+20h] [rbp-90h]
  __int64 v71; // [rsp+20h] [rbp-90h]
  _QWORD *v72; // [rsp+20h] [rbp-90h]
  __int64 v73; // [rsp+20h] [rbp-90h]
  __int64 v74; // [rsp+20h] [rbp-90h]
  _QWORD *v75; // [rsp+20h] [rbp-90h]
  _QWORD *v76; // [rsp+28h] [rbp-88h]
  int v77; // [rsp+28h] [rbp-88h]
  __int64 v78; // [rsp+28h] [rbp-88h]
  _QWORD *v79; // [rsp+28h] [rbp-88h]
  _QWORD *v80; // [rsp+28h] [rbp-88h]
  __int64 v81; // [rsp+28h] [rbp-88h]
  __int64 v82; // [rsp+28h] [rbp-88h]
  int v83; // [rsp+30h] [rbp-80h]
  __int64 v84; // [rsp+30h] [rbp-80h]
  unsigned __int64 v85; // [rsp+30h] [rbp-80h]
  __int64 v86; // [rsp+30h] [rbp-80h]
  __int64 v87; // [rsp+30h] [rbp-80h]
  __int64 v88; // [rsp+38h] [rbp-78h]
  __int64 v90; // [rsp+40h] [rbp-70h]
  __int64 v92; // [rsp+58h] [rbp-58h]
  unsigned __int64 v93; // [rsp+58h] [rbp-58h]
  __m128i v94; // [rsp+60h] [rbp-50h]

  v10 = (__m128i *)sub_2E09D00((__int64 *)a5, a4);
  v11 = *(__m128i **)a5;
  v12 = a6;
  v13 = a2;
  if ( v10 == (__m128i *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8))
    || (*(_DWORD *)((v10->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10->m128i_i64[0] >> 1) & 3) >= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
  {
    v92 = 0;
    if ( v10 != v11 )
      v11 = (__m128i *)((char *)v10 - 24);
  }
  else
  {
    v92 = v10->m128i_i64[1];
    v11 = v10;
  }
LABEL_5:
  while ( a3 != v13 )
  {
    v15 = (_QWORD *)(*(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
    v16 = v15;
    if ( !v15 )
      BUG();
    a3 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
    v17 = *v15;
    if ( (v17 & 4) == 0 && (*((_BYTE *)v16 + 44) & 4) != 0 )
    {
      for ( i = v17; ; i = *(_QWORD *)a3 )
      {
        a3 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(a3 + 44) & 4) == 0 )
          break;
      }
    }
    v19 = *(_WORD *)(a3 + 68);
    if ( (unsigned __int16)(v19 - 14) > 4u && v19 != 24 )
    {
      v20 = a3;
      for ( j = a3; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
        ;
      if ( (*(_DWORD *)(a3 + 44) & 8) != 0 )
      {
        do
          v20 = *(_QWORD *)(v20 + 8);
        while ( (*(_BYTE *)(v20 + 44) & 8) != 0 );
      }
      for ( k = *(_QWORD *)(v20 + 8); k != j; j = *(_QWORD *)(j + 8) )
      {
        v23 = *(_WORD *)(j + 68);
        if ( (unsigned __int16)(v23 - 14) > 4u && v23 != 24 )
          break;
      }
      v24 = a1[4];
      v25 = *(unsigned int *)(v24 + 144);
      v26 = *(_QWORD *)(v24 + 128);
      if ( !(_DWORD)v25 )
        goto LABEL_42;
      v27 = (v25 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( *v28 != j )
      {
        v36 = 1;
        while ( v29 != -4096 )
        {
          v52 = v36 + 1;
          v27 = (v25 - 1) & (v36 + v27);
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( *v28 == j )
            goto LABEL_24;
          v36 = v52;
        }
LABEL_42:
        v28 = (__int64 *)(v26 + 16 * v25);
      }
LABEL_24:
      v30 = *(_QWORD *)(a3 + 32);
      v31 = *(_QWORD *)((v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 16);
      v32 = v28[1] & 0xFFFFFFFFFFFFFFF8LL;
      v90 = *(_QWORD *)((v11->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16);
      v33 = v30 + 40LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF);
      if ( v33 != v30 )
      {
        while ( *(_BYTE *)v30
             || (_DWORD)v12 != *(_DWORD *)(v30 + 8)
             || (*(_OWORD *)(*(_QWORD *)(a1[2] + 272LL) + 16LL * ((*(_DWORD *)v30 >> 8) & 0xFFF)) & a7) == 0 )
        {
LABEL_26:
          v30 += 40;
          if ( v33 == v30 )
            goto LABEL_5;
        }
        if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
        {
          if ( !v90 && (v11->m128i_i8[8] & 6) != 0 )
            v11->m128i_i64[1] = v32 | 4;
          v37 = v32 | 4;
          if ( (v92 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            v37 = v92;
          v92 = v37;
          goto LABEL_26;
        }
        v88 = v32 | 4;
        if ( !v31 )
        {
          if ( ((v11->m128i_i8[8] ^ 6) & 6) != 0 )
          {
            v38 = v32 | 4;
            v39 = v11[1].m128i_i64[0];
            v92 = 0;
            v11->m128i_i64[0] = v88;
            *(_QWORD *)(v39 + 8) = v88;
            if ( (*(_DWORD *)v30 & 0xFFF00) != 0 )
            {
              if ( (*(_BYTE *)(v30 + 4) & 1) != 0 )
                v38 = 0;
              v92 = v38;
            }
            goto LABEL_26;
          }
          v64 = v12;
          v71 = v13;
          v79 = a1;
          v86 = v33;
          v51 = sub_2E0A580(a5, v11->m128i_i8, 1);
          v11 = *(__m128i **)a5;
          v33 = v86;
          a1 = v79;
          v13 = v71;
          v12 = v64;
          if ( v51 != *(char **)a5 )
            v11 = (__m128i *)(v51 - 24);
        }
        if ( (v92 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( v11->m128i_i64[0] == v88 )
          {
LABEL_34:
            v34 = 0;
            v92 = 0;
            if ( (*(_DWORD *)v30 & 0xFFF00) != 0 )
            {
              if ( (*(_BYTE *)(v30 + 4) & 1) == 0 )
                v34 = v32 | 4;
              v92 = v34;
            }
            goto LABEL_26;
          }
          v45 = *(_DWORD *)(a5 + 72);
          a1[17] += 16LL;
          v77 = v45;
          v46 = a1[7];
          v85 = (v46 + 15) & 0xFFFFFFFFFFFFFFF0LL;
          if ( a1[8] >= v85 + 16 && v46 )
          {
            a1[7] = v85 + 16;
            v47 = (v46 + 15) & 0xFFFFFFFFFFFFFFF0LL;
            if ( v85 )
              goto LABEL_67;
          }
          else
          {
            v55 = v12;
            v58 = v13;
            v65 = v33;
            v72 = a1;
            v47 = sub_9D1E70((__int64)(a1 + 7), 16, 16, 4);
            v12 = v55;
            v13 = v58;
            v85 = v47;
            v33 = v65;
            a1 = v72;
LABEL_67:
            *(_DWORD *)v47 = v77;
            *(_QWORD *)(v47 + 8) = v88;
          }
          v48 = *(unsigned int *)(a5 + 72);
          if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 76) )
          {
            v56 = v12;
            v60 = v13;
            v67 = a1;
            v74 = v33;
            v81 = v47;
            sub_C8D5F0(a5 + 64, (const void *)(a5 + 80), v48 + 1, 8u, v33, v12);
            v48 = *(unsigned int *)(a5 + 72);
            v12 = v56;
            v13 = v60;
            a1 = v67;
            v33 = v74;
            v47 = v81;
          }
          v49 = *(_QWORD *)(a5 + 64);
          v57 = v12;
          *(_QWORD *)(v49 + 8 * v48) = v85;
          ++*(_DWORD *)(a5 + 72);
          *((_QWORD *)&v54 + 1) = v92;
          *(_QWORD *)&v54 = v32 | 4;
          v63 = v13;
          v70 = a1;
          v78 = v33;
          v50 = sub_2E0F080(a5, v88, v49, v48, v33, v12, v54, v47);
          v12 = v57;
          v13 = v63;
          a1 = v70;
          v33 = v78;
          v11 = (__m128i *)v50;
          goto LABEL_34;
        }
        v40 = *(_DWORD *)(a5 + 72);
        a1[17] += 16LL;
        v83 = v40;
        v41 = a1[7];
        v93 = (v41 + 15) & 0xFFFFFFFFFFFFFFF0LL;
        if ( a1[8] >= v93 + 16 && v41 )
        {
          a1[7] = v93 + 16;
          v42 = (v41 + 15) & 0xFFFFFFFFFFFFFFF0LL;
          if ( v93 )
            goto LABEL_60;
        }
        else
        {
          v59 = v12;
          v66 = v13;
          v73 = v33;
          v80 = a1;
          v42 = sub_9D1E70((__int64)(a1 + 7), 16, 16, 4);
          v12 = v59;
          v13 = v66;
          v93 = v42;
          v33 = v73;
          a1 = v80;
LABEL_60:
          *(_DWORD *)v42 = v83;
          *(_QWORD *)(v42 + 8) = v88;
        }
        v43 = *(unsigned int *)(a5 + 72);
        if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 76) )
        {
          v61 = v12;
          v68 = v13;
          v75 = a1;
          v82 = v33;
          v87 = v42;
          sub_C8D5F0(a5 + 64, (const void *)(a5 + 80), v43 + 1, 8u, v33, v12);
          v43 = *(unsigned int *)(a5 + 72);
          v42 = v87;
          v12 = v61;
          v13 = v68;
          a1 = v75;
          v33 = v82;
        }
        v62 = v12;
        *(_QWORD *)(*(_QWORD *)(a5 + 64) + 8 * v43) = v93;
        ++*(_DWORD *)(a5 + 72);
        *((_QWORD *)&v53 + 1) = v32 | 6;
        *(_QWORD *)&v53 = v32 | 4;
        v69 = v13;
        v76 = a1;
        v84 = v33;
        v44 = sub_2E0F080(a5, v88, v32 | 6, v43, v33, v12, v53, v42);
        v33 = v84;
        a1 = v76;
        v13 = v69;
        v12 = v62;
        v11 = (__m128i *)v44;
        goto LABEL_34;
      }
    }
  }
  result = v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( !*(_QWORD *)(result + 16) )
  {
    result = v11->m128i_i64[1] ^ 6;
    if ( (result & 6) == 0 )
    {
      v94 = _mm_loadu_si128(v11);
      return (unsigned __int64)sub_2E0C3B0(a5, v94.m128i_i64[0], v94.m128i_i64[1], 1);
    }
  }
  return result;
}
