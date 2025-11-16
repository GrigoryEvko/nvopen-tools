// Function: sub_1036F30
// Address: 0x1036f30
//
__m128i *__fastcall sub_1036F30(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __int64 v10; // rax
  bool v11; // zf
  __m128i *result; // rax
  bool v13; // si
  bool v14; // di
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // r9d
  _QWORD *v19; // r15
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int16 v22; // ax
  __int64 v23; // rax
  __int64 v24; // rdx
  __m128i *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // r9
  int v29; // edx
  __int64 v30; // rax
  __m128i v31; // xmm5
  unsigned __int64 v32; // rsi
  __m128i *v33; // rdx
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  __int64 v37; // rax
  const __m128i *v38; // r9
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // r8
  __m128i *v41; // rax
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  unsigned int v44; // esi
  __int64 v45; // r8
  unsigned int v46; // edi
  unsigned __int64 *v47; // r10
  unsigned __int64 v48; // rcx
  __int64 v49; // rdi
  __int64 *v50; // rax
  int v51; // eax
  int v52; // ecx
  __int64 v53; // rsi
  unsigned int v54; // edx
  int v55; // edi
  unsigned __int64 *v56; // rax
  unsigned __int64 v57; // r8
  _QWORD *v58; // rsi
  _QWORD *v59; // rdx
  _QWORD *v60; // rax
  __int64 v61; // rcx
  int v62; // r15d
  int v63; // r11d
  const void *v64; // rsi
  __int8 *v65; // r12
  const void *v66; // rsi
  int v67; // ecx
  int v68; // eax
  int v69; // ecx
  __int64 v70; // r8
  unsigned __int64 *v71; // r9
  int v72; // r11d
  unsigned int v73; // edx
  unsigned __int64 v74; // rsi
  int v75; // r11d
  char v77; // [rsp+10h] [rbp-1F0h]
  __int64 v78; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 *v79; // [rsp+18h] [rbp-1E8h]
  __int8 *v80; // [rsp+18h] [rbp-1E8h]
  int v81; // [rsp+18h] [rbp-1E8h]
  __m128i v82; // [rsp+20h] [rbp-1E0h] BYREF
  __int64 v83; // [rsp+30h] [rbp-1D0h]
  __m128i v84[3]; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v85[4]; // [rsp+70h] [rbp-190h] BYREF
  _QWORD *v86; // [rsp+90h] [rbp-170h]
  __int64 v87; // [rsp+98h] [rbp-168h]
  _QWORD v88[4]; // [rsp+A0h] [rbp-160h] BYREF
  __m128i v89; // [rsp+C0h] [rbp-140h] BYREF
  __m128i v90; // [rsp+D0h] [rbp-130h] BYREF
  __m128i v91[15]; // [rsp+E0h] [rbp-120h] BYREF
  char v92; // [rsp+1D0h] [rbp-30h] BYREF

  sub_D66840(&v89, (_BYTE *)a2);
  v7 = _mm_loadu_si128(&v89);
  v8 = _mm_loadu_si128(&v90);
  v9 = _mm_loadu_si128(v91);
  v77 = *(_BYTE *)a2;
  v10 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(a3 + 8) = 0;
  v11 = *(_BYTE *)a2 == 61;
  v78 = v10;
  v84[0] = v7;
  v84[1] = v8;
  v84[2] = v9;
  if ( v11 )
  {
    result = (__m128i *)sub_102D1C0(a1 + 872, a2);
    if ( (_BYTE)result )
      return result;
  }
  v90.m128i_i64[0] = a2;
  v89 = 0u;
  v13 = a2 != -8192;
  v14 = a2 != -4096;
  if ( a2 == -8192 || a2 == -4096 )
  {
    v16 = *(unsigned int *)(a1 + 56);
    v17 = *(_QWORD *)(a1 + 40);
    v15 = a2;
    if ( !(_DWORD)v16 )
      goto LABEL_29;
  }
  else
  {
    sub_BD73F0((__int64)&v89);
    v15 = v90.m128i_i64[0];
    v16 = *(unsigned int *)(a1 + 56);
    v17 = *(_QWORD *)(a1 + 40);
    v13 = v90.m128i_i64[0] != -8192;
    v14 = v90.m128i_i64[0] != -4096;
    if ( !(_DWORD)v16 )
      goto LABEL_29;
  }
  v18 = (v16 - 1) & (((unsigned int)v15 >> 4) ^ ((unsigned int)v15 >> 9));
  v19 = (_QWORD *)(v17 + 48LL * v18);
  v20 = v19[2];
  if ( v15 != v20 )
  {
    v62 = 1;
    while ( v20 != -4096 )
    {
      v63 = v62 + 1;
      v18 = (v16 - 1) & (v62 + v18);
      v19 = (_QWORD *)(v17 + 48LL * v18);
      v20 = v19[2];
      if ( v20 == v15 )
        goto LABEL_7;
      v62 = v63;
    }
LABEL_29:
    v19 = (_QWORD *)(v17 + 48 * v16);
    if ( !v13 || !v14 || !v15 )
      goto LABEL_11;
    goto LABEL_9;
  }
LABEL_7:
  if ( v15 && v13 && v14 )
  {
LABEL_9:
    sub_BD60C0(&v89);
    v17 = *(_QWORD *)(a1 + 40);
    v16 = *(unsigned int *)(a1 + 56);
  }
  if ( v19 == (_QWORD *)(48 * v16 + v17) )
  {
LABEL_11:
    if ( sub_B46560((unsigned __int8 *)a2)
      || (*(_BYTE *)a2 == 61 || *(_BYTE *)a2 == 62)
      && ((v22 = *(_WORD *)(a2 + 2), ((v22 >> 7) & 6) != 0) || (v22 & 1) != 0) )
    {
      v32 = *(unsigned int *)(a3 + 12);
      v33 = &v89;
      v34 = *(_QWORD *)a3;
      v89.m128i_i64[0] = v78;
      v89.m128i_i64[1] = 0x6000000000000003LL;
      v90.m128i_i64[0] = v84[0].m128i_i64[0];
      v35 = *(unsigned int *)(a3 + 8);
      v36 = v35 + 1;
      if ( v35 + 1 > v32 )
      {
        v64 = (const void *)(a3 + 16);
        if ( v34 > (unsigned __int64)&v89 || (unsigned __int64)&v89 >= v34 + 24 * v35 )
        {
          sub_C8D5F0(a3, v64, v36, 0x18u, v36, v21);
          v34 = *(_QWORD *)a3;
          v35 = *(unsigned int *)(a3 + 8);
          v33 = &v89;
        }
        else
        {
          v65 = &v89.m128i_i8[-v34];
          sub_C8D5F0(a3, v64, v36, 0x18u, v36, v21);
          v34 = *(_QWORD *)a3;
          v35 = *(unsigned int *)(a3 + 8);
          v33 = (__m128i *)&v65[*(_QWORD *)a3];
        }
      }
      result = (__m128i *)(v34 + 24 * v35);
      *result = _mm_loadu_si128(v33);
      result[1].m128i_i64[0] = v33[1].m128i_i64[0];
      ++*(_DWORD *)(a3 + 8);
    }
    else
    {
      v23 = sub_AA4E30(v78);
      v24 = *(_QWORD *)(a1 + 264);
      v85[2] = 0;
      v86 = v88;
      v85[1] = v23;
      v85[0] = v84[0].m128i_i64[0];
      v85[3] = v24;
      v87 = 0x400000000LL;
      if ( *(_BYTE *)v84[0].m128i_i64[0] > 0x1Cu )
      {
        v88[0] = v84[0].m128i_i64[0];
        LODWORD(v87) = 1;
      }
      v89.m128i_i64[0] = 0;
      v25 = &v90;
      v89.m128i_i64[1] = 1;
      do
      {
        v25->m128i_i64[0] = -4096;
        ++v25;
      }
      while ( v25 != (__m128i *)&v92 );
      v26 = a2;
      result = (__m128i *)sub_1035170(a1, (_BYTE *)a2, v85, v84, v77 == 61, v78, a3, (__int64)&v89, 1u, a4, 0);
      if ( !(_BYTE)result )
      {
        v29 = *(_DWORD *)(a3 + 12);
        *(_DWORD *)(a3 + 8) = 0;
        v82.m128i_i64[0] = v78;
        v82.m128i_i64[1] = 0x6000000000000003LL;
        v83 = v84[0].m128i_i64[0];
        v30 = 0;
        if ( !v29 )
        {
          v26 = a3 + 16;
          sub_C8D5F0(a3, (const void *)(a3 + 16), 1u, 0x18u, v27, v28);
          v30 = 24LL * *(unsigned int *)(a3 + 8);
        }
        v31 = _mm_loadu_si128(&v82);
        result = (__m128i *)(*(_QWORD *)a3 + v30);
        result[1].m128i_i64[0] = v83;
        *result = v31;
        ++*(_DWORD *)(a3 + 8);
      }
      if ( (v89.m128i_i8[8] & 1) == 0 )
      {
        v26 = 16LL * v90.m128i_u32[2];
        result = (__m128i *)sub_C7D6A0(v90.m128i_i64[0], v26, 8);
      }
      if ( v86 != v88 )
        return (__m128i *)_libc_free(v86, v26);
    }
    return result;
  }
  v37 = *(unsigned int *)(a3 + 8);
  v38 = (const __m128i *)(v19 + 3);
  v39 = *(_QWORD *)a3;
  v40 = v37 + 1;
  if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v66 = (const void *)(a3 + 16);
    if ( v39 > (unsigned __int64)v38 || (unsigned __int64)v38 >= v39 + 24 * v37 )
    {
      sub_C8D5F0(a3, v66, v40, 0x18u, v40, (__int64)v38);
      v39 = *(_QWORD *)a3;
      v37 = *(unsigned int *)(a3 + 8);
      v38 = (const __m128i *)(v19 + 3);
    }
    else
    {
      v80 = &v38->m128i_i8[-v39];
      sub_C8D5F0(a3, v66, v40, 0x18u, v40, (__int64)v38->m128i_i64 - v39);
      v39 = *(_QWORD *)a3;
      v37 = *(unsigned int *)(a3 + 8);
      v38 = (const __m128i *)&v80[*(_QWORD *)a3];
    }
  }
  v41 = (__m128i *)(v39 + 24 * v37);
  *v41 = _mm_loadu_si128(v38);
  v41[1].m128i_i64[0] = v38[1].m128i_i64[0];
  ++*(_DWORD *)(a3 + 8);
  v42 = v19[4] & 0xFFFFFFFFFFFFFFF8LL;
  v43 = v19[4] & 7LL;
  if ( (unsigned int)v43 <= 2 )
  {
    v44 = *(_DWORD *)(a1 + 88);
    if ( v44 )
      goto LABEL_37;
LABEL_51:
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_52;
  }
  if ( (_DWORD)v43 != 3 )
    BUG();
  v44 = *(_DWORD *)(a1 + 88);
  v42 = 0;
  if ( !v44 )
    goto LABEL_51;
LABEL_37:
  v45 = *(_QWORD *)(a1 + 72);
  v46 = (v44 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
  v47 = (unsigned __int64 *)(v45 + 72LL * v46);
  v48 = *v47;
  if ( v42 != *v47 )
  {
    v81 = 1;
    v56 = 0;
    while ( v48 != -4096 )
    {
      if ( v48 == -8192 && !v56 )
        v56 = v47;
      v46 = (v44 - 1) & (v81 + v46);
      v47 = (unsigned __int64 *)(v45 + 72LL * v46);
      v48 = *v47;
      if ( v42 == *v47 )
        goto LABEL_38;
      ++v81;
    }
    v67 = *(_DWORD *)(a1 + 80);
    if ( !v56 )
      v56 = v47;
    ++*(_QWORD *)(a1 + 64);
    v55 = v67 + 1;
    if ( 4 * (v67 + 1) < 3 * v44 )
    {
      if ( v44 - *(_DWORD *)(a1 + 84) - v55 > v44 >> 3 )
        goto LABEL_54;
      sub_102F2A0(a1 + 64, v44);
      v68 = *(_DWORD *)(a1 + 88);
      if ( v68 )
      {
        v69 = v68 - 1;
        v70 = *(_QWORD *)(a1 + 72);
        v71 = 0;
        v72 = 1;
        v73 = (v68 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v55 = *(_DWORD *)(a1 + 80) + 1;
        v56 = (unsigned __int64 *)(v70 + 72LL * v73);
        v74 = *v56;
        if ( v42 != *v56 )
        {
          while ( v74 != -4096 )
          {
            if ( !v71 && v74 == -8192 )
              v71 = v56;
            v73 = v69 & (v72 + v73);
            v56 = (unsigned __int64 *)(v70 + 72LL * v73);
            v74 = *v56;
            if ( v42 == *v56 )
              goto LABEL_54;
            ++v72;
          }
LABEL_85:
          if ( v71 )
            v56 = v71;
          goto LABEL_54;
        }
        goto LABEL_54;
      }
      goto LABEL_105;
    }
LABEL_52:
    sub_102F2A0(a1 + 64, 2 * v44);
    v51 = *(_DWORD *)(a1 + 88);
    if ( v51 )
    {
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 72);
      v54 = (v51 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
      v55 = *(_DWORD *)(a1 + 80) + 1;
      v56 = (unsigned __int64 *)(v53 + 72LL * v54);
      v57 = *v56;
      if ( v42 != *v56 )
      {
        v75 = 1;
        v71 = 0;
        while ( v57 != -4096 )
        {
          if ( !v71 && v57 == -8192 )
            v71 = v56;
          v54 = v52 & (v75 + v54);
          v56 = (unsigned __int64 *)(v53 + 72LL * v54);
          v57 = *v56;
          if ( v42 == *v56 )
            goto LABEL_54;
          ++v75;
        }
        goto LABEL_85;
      }
LABEL_54:
      *(_DWORD *)(a1 + 80) = v55;
      if ( *v56 != -4096 )
        --*(_DWORD *)(a1 + 84);
      *v56 = v42;
      v49 = (__int64)(v56 + 1);
      v56[1] = 0;
      v56[2] = (unsigned __int64)(v56 + 5);
      v56[3] = 4;
      *((_DWORD *)v56 + 8) = 0;
      *((_BYTE *)v56 + 36) = 1;
LABEL_57:
      v58 = *(_QWORD **)(v49 + 8);
      v59 = &v58[*(unsigned int *)(v49 + 20)];
      v60 = v58;
      if ( v58 != v59 )
      {
        while ( a2 != *v60 )
        {
          if ( v59 == ++v60 )
            goto LABEL_41;
        }
        v61 = (unsigned int)(*(_DWORD *)(v49 + 20) - 1);
        *(_DWORD *)(v49 + 20) = v61;
        *v60 = v58[v61];
        ++*(_QWORD *)v49;
      }
      goto LABEL_41;
    }
LABEL_105:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
LABEL_38:
  v49 = (__int64)(v47 + 1);
  if ( *((_BYTE *)v47 + 36) )
    goto LABEL_57;
  v79 = v47;
  v50 = sub_C8CA60(v49, a2);
  if ( v50 )
  {
    *v50 = -2;
    ++*((_DWORD *)v79 + 8);
    ++v79[1];
  }
LABEL_41:
  v89 = 0u;
  v90.m128i_i64[0] = -8192;
  result = (__m128i *)v19[2];
  if ( result != (__m128i *)-8192LL )
  {
    if ( result != (__m128i *)-4096LL && result )
      sub_BD60C0(v19);
    v19[2] = -8192;
    result = (__m128i *)v90.m128i_i64[0];
    if ( v90.m128i_i64[0] != -4096 && v90.m128i_i64[0] != 0 && v90.m128i_i64[0] != -8192 )
      result = (__m128i *)sub_BD60C0(&v89);
  }
  --*(_DWORD *)(a1 + 48);
  ++*(_DWORD *)(a1 + 52);
  return result;
}
