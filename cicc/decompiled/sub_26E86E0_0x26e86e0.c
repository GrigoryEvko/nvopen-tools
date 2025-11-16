// Function: sub_26E86E0
// Address: 0x26e86e0
//
void __fastcall sub_26E86E0(unsigned __int64 *a1, __int64 m128i_i64)
{
  const __m128i *v2; // rbx
  size_t v3; // rdx
  int *v4; // rsi
  size_t v5; // r13
  unsigned __int64 v6; // rcx
  __m128i **v7; // rax
  __m128i *v8; // r12
  __int64 v9; // r15
  unsigned __int64 v10; // r13
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  void *v16; // rax
  unsigned __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rcx
  unsigned __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r12
  _QWORD *v26; // rax
  __int64 v27; // r11
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 *v31; // rax
  __int64 *v32; // rbx
  unsigned __int64 v33; // r14
  __int64 v34; // r13
  __int64 v35; // r14
  __int64 v36; // rcx
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // r15
  __int64 v40; // rsi
  unsigned int v41; // ecx
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rdi
  unsigned int v45; // r10d
  __int64 v46; // rax
  __int64 v47; // r11
  int v48; // eax
  int v49; // eax
  int v50; // ebx
  int v51; // eax
  int v52; // eax
  int v53; // ebx
  _QWORD *v54; // rax
  unsigned __int64 v55; // r15
  __m128i v56; // xmm0
  __m128i v57; // xmm1
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  __int32 v62; // ecx
  __int64 v63; // rdx
  unsigned __int64 v64; // rax
  __int32 v65; // ecx
  unsigned __int64 v66; // r9
  __m128i **v67; // rax
  unsigned __int64 v68; // r13
  unsigned __int64 v69; // r12
  _QWORD *v70; // rdi
  unsigned __int64 v71; // [rsp+8h] [rbp-1E8h]
  char v72; // [rsp+17h] [rbp-1D9h]
  char v73; // [rsp+38h] [rbp-1B8h]
  unsigned __int64 v74; // [rsp+38h] [rbp-1B8h]
  __m128i *v75; // [rsp+38h] [rbp-1B8h]
  __int64 v77; // [rsp+48h] [rbp-1A8h]
  __int64 v78; // [rsp+48h] [rbp-1A8h]
  __int64 v79; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 *v80; // [rsp+58h] [rbp-198h]
  unsigned __int64 v81; // [rsp+60h] [rbp-190h] BYREF
  int v82[40]; // [rsp+70h] [rbp-180h] BYREF
  __m128i src[14]; // [rsp+110h] [rbp-E0h] BYREF

  v80 = a1 + 5;
  v72 = unk_4F838D3;
  v2 = *(const __m128i **)(a1[1] + 24);
  if ( unk_4F838D3 )
  {
    for ( ; v2; v2 = (const __m128i *)v2->m128i_i64[0] )
    {
      v3 = v2[2].m128i_u64[1];
      v4 = (int *)v2[2].m128i_i64[0];
      v5 = v3;
      memset(src, 0, 0xB0u);
      src[7].m128i_i64[0] = 0;
      v6 = v3;
      v77 = v3;
      src[6].m128i_i64[0] = (__int64)src[5].m128i_i64;
      src[6].m128i_i64[1] = (__int64)src[5].m128i_i64;
      src[8].m128i_i64[1] = 0;
      src[9].m128i_i64[0] = (__int64)src[8].m128i_i64;
      src[9].m128i_i64[1] = (__int64)src[8].m128i_i64;
      src[10].m128i_i64[0] = 0;
      if ( v4 )
      {
        sub_C7D030(v82);
        sub_C7D280(v82, v4, v5);
        sub_C7D290(v82, &v81);
        v6 = v81;
      }
      *(_QWORD *)v82 = v6;
      v7 = (__m128i **)sub_C1DD00(v80, v6 % a1[6], v82, v6);
      if ( v7 )
      {
        v8 = *v7;
        if ( *v7 )
        {
          v9 = src[8].m128i_i64[1];
          v73 = 0;
          if ( !src[8].m128i_i64[1] )
          {
            sub_26DFC60((_QWORD *)src[5].m128i_i64[1]);
            goto LABEL_11;
          }
          do
          {
LABEL_8:
            v10 = v9;
            sub_26E0470(*(_QWORD **)(v9 + 24));
            v11 = *(_QWORD **)(v9 + 56);
            v9 = *(_QWORD *)(v9 + 16);
            sub_26E06C0(v11);
            j_j___libc_free_0(v10);
          }
          while ( v9 );
          sub_26DFC60((_QWORD *)src[5].m128i_i64[1]);
          if ( !v73 )
            goto LABEL_11;
          goto LABEL_10;
        }
      }
      v54 = (_QWORD *)sub_22077B0(0xC8u);
      v55 = (unsigned __int64)v54;
      if ( v54 )
        *v54 = 0;
      v56 = _mm_loadu_si128(&src[1]);
      v57 = _mm_loadu_si128(&src[2]);
      v58 = src[5].m128i_i64[1];
      v54[1] = *(_QWORD *)v82;
      v59 = src[0].m128i_i64[0];
      *(__m128i *)(v55 + 32) = v56;
      *(_QWORD *)(v55 + 16) = v59;
      v60 = src[0].m128i_i64[1];
      *(__m128i *)(v55 + 48) = v57;
      *(_QWORD *)(v55 + 24) = v60;
      *(__m128i *)(v55 + 64) = src[3];
      *(_QWORD *)(v55 + 80) = src[4].m128i_i64[0];
      v61 = v55 + 96;
      if ( v58 )
      {
        v62 = src[5].m128i_i32[0];
        *(_QWORD *)(v55 + 104) = v58;
        *(_DWORD *)(v55 + 96) = v62;
        *(__m128i *)(v55 + 112) = src[6];
        *(_QWORD *)(v58 + 8) = v61;
        v63 = src[8].m128i_i64[1];
        src[5].m128i_i64[1] = 0;
        *(_QWORD *)(v55 + 128) = src[7].m128i_i64[0];
        src[7].m128i_i64[0] = 0;
        src[6].m128i_i64[0] = (__int64)src[5].m128i_i64;
        src[6].m128i_i64[1] = (__int64)src[5].m128i_i64;
        v64 = v55 + 144;
        if ( !v63 )
          goto LABEL_102;
      }
      else
      {
        v63 = src[8].m128i_i64[1];
        *(_QWORD *)(v55 + 112) = v61;
        *(_QWORD *)(v55 + 120) = v61;
        v64 = v55 + 144;
        *(_DWORD *)(v55 + 96) = 0;
        *(_QWORD *)(v55 + 104) = 0;
        *(_QWORD *)(v55 + 128) = 0;
        if ( !v63 )
        {
LABEL_102:
          *(_DWORD *)(v55 + 144) = 0;
          *(_QWORD *)(v55 + 152) = 0;
          *(_QWORD *)(v55 + 160) = v64;
          *(_QWORD *)(v55 + 168) = v64;
          *(_QWORD *)(v55 + 176) = 0;
          goto LABEL_90;
        }
      }
      v65 = src[8].m128i_i32[0];
      *(_QWORD *)(v55 + 152) = v63;
      *(_DWORD *)(v55 + 144) = v65;
      *(__m128i *)(v55 + 160) = src[9];
      *(_QWORD *)(v63 + 8) = v64;
      src[8].m128i_i64[1] = 0;
      *(_QWORD *)(v55 + 176) = src[10].m128i_i64[0];
      src[10].m128i_i64[0] = 0;
      src[9].m128i_i64[0] = (__int64)src[8].m128i_i64;
      src[9].m128i_i64[1] = (__int64)src[8].m128i_i64;
LABEL_90:
      v66 = *(_QWORD *)(v55 + 8);
      *(_QWORD *)(v55 + 184) = src[10].m128i_i64[1];
      v71 = v66;
      v74 = v66 % a1[6];
      v67 = (__m128i **)sub_C1DD00(v80, v74, (_QWORD *)(v55 + 8), v66);
      if ( v67 && (v8 = *v67) != 0 )
      {
        if ( *(_QWORD *)(v55 + 152) )
        {
          v75 = *v67;
          v68 = *(_QWORD *)(v55 + 152);
          do
          {
            v69 = v68;
            sub_26E0470(*(_QWORD **)(v68 + 24));
            v70 = *(_QWORD **)(v68 + 56);
            v68 = *(_QWORD *)(v68 + 16);
            sub_26E06C0(v70);
            j_j___libc_free_0(v69);
          }
          while ( v68 );
          v8 = v75;
        }
        sub_26DFC60(*(_QWORD **)(v55 + 104));
        j_j___libc_free_0(v55);
      }
      else
      {
        v8 = (__m128i *)sub_26E00B0(v80, v74, v71, (_QWORD *)v55, 1);
      }
      v9 = src[8].m128i_i64[1];
      v73 = v72;
      if ( src[8].m128i_i64[1] )
        goto LABEL_8;
      sub_26DFC60((_QWORD *)src[5].m128i_i64[1]);
LABEL_10:
      v8[3].m128i_i64[0] = 0;
      v8[3].m128i_i64[1] = 0;
      v8[2].m128i_i64[0] = (__int64)v4;
      v8[4].m128i_i64[0] = 0;
      v8[2].m128i_i64[1] = v77;
LABEL_11:
      m128i_i64 = (__int64)v2[1].m128i_i64;
      sub_C1D5C0(v8 + 1, v2 + 1, 1u);
    }
LABEL_12:
    if ( !LOBYTE(qword_4FF8120[17]) )
      goto LABEL_13;
    goto LABEL_82;
  }
  if ( !v2 )
    goto LABEL_12;
  do
  {
    m128i_i64 = (__int64)v2[1].m128i_i64;
    sub_EFA6B0(v80, (__int64)v2[1].m128i_i64);
    v2 = (const __m128i *)v2->m128i_i64[0];
  }
  while ( v2 );
  if ( LOBYTE(qword_4FF8120[17]) )
LABEL_82:
    sub_26E3760(a1);
LABEL_13:
  memset(src, 0, 24);
  v12 = *a1 + 24;
  v13 = *(_QWORD *)(*a1 + 32);
  if ( v13 == v12 )
    goto LABEL_20;
  v14 = 0;
  do
  {
    v13 = *(_QWORD *)(v13 + 8);
    ++v14;
  }
  while ( v12 != v13 );
  if ( v14 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v15 = 8 * v14;
  v16 = (void *)sub_22077B0(8 * v14);
  v17 = src[0].m128i_i64[0];
  v18 = (__int64)v16;
  if ( src[0].m128i_i64[1] - src[0].m128i_i64[0] > 0 )
  {
    memmove(v16, (const void *)src[0].m128i_i64[0], src[0].m128i_i64[1] - src[0].m128i_i64[0]);
    m128i_i64 = src[1].m128i_i64[0] - v17;
LABEL_84:
    j_j___libc_free_0(v17);
    goto LABEL_19;
  }
  if ( src[0].m128i_i64[0] )
  {
    m128i_i64 = src[1].m128i_i64[0] - src[0].m128i_i64[0];
    goto LABEL_84;
  }
LABEL_19:
  src[0].m128i_i64[0] = v18;
  src[0].m128i_i64[1] = v18;
  src[1].m128i_i64[0] = v18 + v15;
LABEL_20:
  v19 = a1[2];
  v79 = v19;
  sub_D2AD40(v19, (__int64 *)m128i_i64);
  v23 = *(unsigned int *)(v19 + 440);
  v24 = v19;
  if ( !(_DWORD)v23 )
    goto LABEL_29;
  v20 = *(_QWORD *)(v19 + 432);
  v25 = *(_QWORD *)v20;
  if ( !*(_QWORD *)v20 )
    goto LABEL_29;
  while ( 1 )
  {
    v29 = *(unsigned int *)(v25 + 16);
    if ( (_DWORD)v29 )
      break;
    v22 = *(unsigned int *)(v24 + 600);
    v21 = *(_QWORD *)(v24 + 584);
    if ( (_DWORD)v22 )
    {
      m128i_i64 = ((_DWORD)v22 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v26 = (_QWORD *)(v21 + 16 * m128i_i64);
      v27 = *v26;
      if ( *v26 == v25 )
      {
LABEL_24:
        v28 = *((_DWORD *)v26 + 2) + 1;
        if ( v28 == (_DWORD)v23 )
          goto LABEL_29;
        goto LABEL_25;
      }
      v52 = 1;
      while ( v27 != -4096 )
      {
        v53 = v52 + 1;
        m128i_i64 = ((_DWORD)v22 - 1) & (unsigned int)(v52 + m128i_i64);
        v26 = (_QWORD *)(v21 + 16LL * (unsigned int)m128i_i64);
        v27 = *v26;
        if ( *v26 == v25 )
          goto LABEL_24;
        v52 = v53;
      }
    }
    v28 = *(_DWORD *)(v21 + 16LL * (unsigned int)v22 + 8) + 1;
    if ( v28 == (_DWORD)v23 )
      goto LABEL_29;
LABEL_25:
    v25 = *(_QWORD *)(v20 + 8LL * v28);
    if ( !v25 )
      goto LABEL_29;
  }
LABEL_45:
  v34 = *(_QWORD *)(v25 + 8);
  v78 = v25;
  v35 = v34 + 8 * v29;
  do
  {
    v36 = *(_QWORD *)(*(_QWORD *)v34 + 8LL);
    v37 = v36 + 8LL * *(unsigned int *)(*(_QWORD *)v34 + 16LL);
    v38 = v36;
    if ( v36 != v37 )
    {
      while ( 1 )
      {
        v39 = *(_QWORD *)(*(_QWORD *)v38 + 8LL);
        if ( sub_B2FC80(v39) || !(unsigned __int8)sub_B2D620(v39, "use-sample-profile", 0x12u) )
          goto LABEL_48;
        *(_QWORD *)v82 = v39;
        v40 = src[0].m128i_i64[1];
        if ( src[0].m128i_i64[1] == src[1].m128i_i64[0] )
        {
          sub_24147A0((__int64)src, (_BYTE *)src[0].m128i_i64[1], v82);
LABEL_48:
          v38 += 8;
          if ( v37 == v38 )
            break;
        }
        else
        {
          if ( src[0].m128i_i64[1] )
          {
            *(_QWORD *)src[0].m128i_i64[1] = v39;
            v40 = src[0].m128i_i64[1];
          }
          v38 += 8;
          src[0].m128i_i64[1] = v40 + 8;
          if ( v37 == v38 )
            break;
        }
      }
    }
    v34 += 8;
  }
  while ( v35 != v34 );
  v23 = *(unsigned int *)(v79 + 600);
  m128i_i64 = *(_QWORD *)(v79 + 584);
  if ( !(_DWORD)v23 )
  {
LABEL_74:
    v42 = m128i_i64 + 16LL * (unsigned int)v23;
    goto LABEL_58;
  }
  v41 = (v23 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
  v42 = m128i_i64 + 16LL * v41;
  v21 = *(_QWORD *)v42;
  if ( *(_QWORD *)v42 != v78 )
  {
    v51 = 1;
    while ( v21 != -4096 )
    {
      v22 = (unsigned int)(v51 + 1);
      v41 = (v23 - 1) & (v51 + v41);
      v42 = m128i_i64 + 16LL * v41;
      v21 = *(_QWORD *)v42;
      if ( *(_QWORD *)v42 == v78 )
        goto LABEL_58;
      v51 = v22;
    }
    goto LABEL_74;
  }
LABEL_58:
  v20 = *(unsigned int *)(v79 + 440);
  v43 = *(_DWORD *)(v42 + 8) + 1;
  if ( v43 != (_DWORD)v20 )
  {
    v44 = *(_QWORD *)(v79 + 432);
    v25 = *(_QWORD *)(v44 + 8LL * v43);
    if ( v25 )
    {
      v21 = (unsigned int)(v23 - 1);
      v22 = m128i_i64 + 16LL * (unsigned int)v23;
      while ( 1 )
      {
        v29 = *(unsigned int *)(v25 + 16);
        if ( (_DWORD)v29 )
          goto LABEL_45;
        if ( (_DWORD)v23 )
        {
          v45 = v21 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v46 = m128i_i64 + 16LL * v45;
          v47 = *(_QWORD *)v46;
          if ( v25 == *(_QWORD *)v46 )
            goto LABEL_62;
          v49 = 1;
          while ( v47 != -4096 )
          {
            v50 = v49 + 1;
            v45 = v21 & (v49 + v45);
            v46 = m128i_i64 + 16LL * v45;
            v47 = *(_QWORD *)v46;
            if ( v25 == *(_QWORD *)v46 )
              goto LABEL_62;
            v49 = v50;
          }
        }
        v46 = m128i_i64 + 16LL * (unsigned int)v23;
LABEL_62:
        v48 = *(_DWORD *)(v46 + 8) + 1;
        if ( (_DWORD)v20 != v48 )
        {
          v25 = *(_QWORD *)(v44 + 8LL * v48);
          if ( v25 )
            continue;
        }
        break;
      }
    }
  }
LABEL_29:
  v30 = src[0].m128i_i64[1];
  v31 = (__int64 *)src[0].m128i_i64[0];
  if ( src[0].m128i_i64[1] != src[0].m128i_i64[0] )
  {
    v23 = src[0].m128i_i64[1] - 8;
    if ( src[0].m128i_i64[0] >= (unsigned __int64)(src[0].m128i_i64[1] - 8) )
      goto LABEL_33;
    do
    {
      v20 = *v31;
      m128i_i64 = *(_QWORD *)v23;
      ++v31;
      v23 -= 8LL;
      *(v31 - 1) = m128i_i64;
      *(_QWORD *)(v23 + 8) = v20;
    }
    while ( v23 > (unsigned __int64)v31 );
    v30 = src[0].m128i_i64[1];
    v31 = (__int64 *)src[0].m128i_i64[0];
    if ( src[0].m128i_i64[1] != src[0].m128i_i64[0] )
    {
LABEL_33:
      v32 = v31;
      do
      {
        while ( 1 )
        {
          v33 = *v32;
          if ( !sub_B2FC80(*v32) )
          {
            m128i_i64 = (__int64)"use-sample-profile";
            if ( (unsigned __int8)sub_B2D620(v33, "use-sample-profile", 0x12u) )
              break;
          }
          if ( (__int64 *)v30 == ++v32 )
            goto LABEL_38;
        }
        m128i_i64 = v33;
        ++v32;
        sub_26E6000((__int64)a1, v33);
      }
      while ( (__int64 *)v30 != v32 );
    }
  }
LABEL_38:
  if ( LOBYTE(qword_4FF8120[17]) )
    sub_26E7C00((__int64)a1);
  if ( LOBYTE(qword_4FF8200[17]) )
    sub_26E11A0((__int64)a1);
  sub_26E4280((__int64)a1, m128i_i64, v23, v20, (_QWORD *)v21, (_QWORD *)v22);
  if ( src[0].m128i_i64[0] )
    j_j___libc_free_0(src[0].m128i_u64[0]);
}
