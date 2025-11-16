// Function: sub_26E6F90
// Address: 0x26e6f90
//
__int64 __fastcall sub_26E6F90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int64 k; // rdx
  unsigned int v13; // r15d
  __int64 v14; // rdx
  __int128 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r9
  const __m128i *v24; // rdi
  bool v25; // al
  size_t v26; // rdx
  size_t v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdi
  _QWORD *v31; // rcx
  int v32; // r8d
  _QWORD *v33; // rax
  __int64 v34; // r10
  int v35; // eax
  int v36; // r9d
  __int64 v37; // r8
  int v38; // r11d
  unsigned int i; // ecx
  const void *v40; // rsi
  unsigned int v41; // ecx
  float v42; // xmm0_4
  signed __int64 v43; // rax
  float v44; // xmm1_4
  _QWORD *v45; // rbx
  unsigned __int64 v46; // rdi
  int v47; // eax
  int v48; // r11d
  int v49; // r8d
  unsigned int j; // r9d
  __int64 v51; // rcx
  const void *v52; // rsi
  unsigned int v53; // r9d
  int v54; // eax
  int v55; // esi
  int v56; // eax
  int v57; // r9d
  int v58; // r11d
  unsigned int v59; // ecx
  const void *v60; // rsi
  unsigned int v61; // ecx
  unsigned __int64 v62; // rdx
  int v63; // edi
  int v64; // r10d
  __int64 v65; // rdi
  int v66; // eax
  bool v67; // al
  int v68; // eax
  bool v69; // al
  int v70; // [rsp+8h] [rbp-1B8h]
  int v71; // [rsp+8h] [rbp-1B8h]
  __int64 v72; // [rsp+10h] [rbp-1B0h]
  __int64 v73; // [rsp+10h] [rbp-1B0h]
  __int64 v74; // [rsp+10h] [rbp-1B0h]
  __int64 v75; // [rsp+18h] [rbp-1A8h]
  unsigned int v76; // [rsp+18h] [rbp-1A8h]
  unsigned int v77; // [rsp+18h] [rbp-1A8h]
  int v78; // [rsp+24h] [rbp-19Ch]
  int v79; // [rsp+24h] [rbp-19Ch]
  int v80; // [rsp+24h] [rbp-19Ch]
  unsigned int v81; // [rsp+28h] [rbp-198h]
  __int64 v82; // [rsp+28h] [rbp-198h]
  __int64 v83; // [rsp+28h] [rbp-198h]
  __int64 v84; // [rsp+30h] [rbp-190h]
  __int64 v85; // [rsp+30h] [rbp-190h]
  __int64 v86; // [rsp+30h] [rbp-190h]
  __int64 v87; // [rsp+38h] [rbp-188h]
  int v88; // [rsp+38h] [rbp-188h]
  int v89; // [rsp+38h] [rbp-188h]
  int v90; // [rsp+38h] [rbp-188h]
  int v91; // [rsp+38h] [rbp-188h]
  int *s1; // [rsp+40h] [rbp-180h]
  void *s1b; // [rsp+40h] [rbp-180h]
  int *s1a; // [rsp+40h] [rbp-180h]
  __int64 v95; // [rsp+48h] [rbp-178h]
  const __m128i *v96; // [rsp+50h] [rbp-170h] BYREF
  __int64 v97; // [rsp+58h] [rbp-168h]
  __int64 v98; // [rsp+60h] [rbp-160h]
  const __m128i *v99; // [rsp+70h] [rbp-150h] BYREF
  __int64 v100; // [rsp+78h] [rbp-148h]
  __int64 v101; // [rsp+80h] [rbp-140h]
  char v102[8]; // [rsp+90h] [rbp-130h] BYREF
  int v103; // [rsp+98h] [rbp-128h] BYREF
  unsigned __int64 v104; // [rsp+A0h] [rbp-120h]
  int *v105; // [rsp+A8h] [rbp-118h]
  int *v106; // [rsp+B0h] [rbp-110h]
  __int64 v107; // [rsp+B8h] [rbp-108h]
  unsigned __int64 v108; // [rsp+C0h] [rbp-100h] BYREF
  int v109; // [rsp+C8h] [rbp-F8h] BYREF
  unsigned __int64 v110; // [rsp+D0h] [rbp-F0h]
  int *v111; // [rsp+D8h] [rbp-E8h]
  int *v112; // [rsp+E0h] [rbp-E0h]
  __int64 v113; // [rsp+E8h] [rbp-D8h]
  __m128i s; // [rsp+F0h] [rbp-D0h] BYREF
  _QWORD *v115; // [rsp+100h] [rbp-C0h]
  __int64 v116; // [rsp+108h] [rbp-B8h]
  char v117; // [rsp+120h] [rbp-A0h] BYREF

  v6 = *(_QWORD *)(a3 + 8);
  s1 = *(int **)a3;
  if ( *(_QWORD *)a3 )
  {
    sub_C7D030(&s);
    sub_C7D280(s.m128i_i32, s1, v6);
    sub_C7D290(&s, &v108);
    v6 = v108;
  }
  v7 = v6 % *(_QWORD *)(a1 + 48);
  s.m128i_i64[0] = v6;
  v8 = sub_C1DD00((_QWORD *)(a1 + 40), v7, &s, v6);
  if ( v8 )
  {
    v9 = *v8;
    if ( v9 )
    {
      v10 = v9 + 16;
      goto LABEL_6;
    }
  }
  if ( !(_BYTE)qword_4FF8648 )
    return 0;
  v28 = 0;
  s1a = *(int **)a3;
  if ( *(_QWORD *)a3 )
    v28 = *(_QWORD *)(a3 + 8);
  s.m128i_i64[0] = 0;
  LODWORD(v116) = 4;
  v29 = sub_C7D670(64, 8);
  v115 = 0;
  v30 = v29;
  v84 = v29;
  v31 = (_QWORD *)v29;
  s.m128i_i64[1] = v29;
  v32 = v116;
  v33 = (_QWORD *)(v29 + 16LL * (unsigned int)v116);
  if ( (_QWORD *)v30 != v33 )
  {
    do
    {
      if ( v31 )
      {
        *v31 = -1;
        v31[1] = 0;
      }
      v31 += 2;
    }
    while ( v33 != v31 );
  }
  if ( !v32 )
  {
    ++s.m128i_i64[0];
LABEL_39:
    sub_BA8070((__int64)&s, 2 * v32);
    v34 = 0;
    v88 = v116;
    if ( !(_DWORD)v116 )
      goto LABEL_69;
    v85 = s.m128i_i64[1];
    v35 = sub_C94890(s1a, v28);
    v36 = 1;
    v37 = 0;
    v38 = v88 - 1;
    for ( i = (v88 - 1) & v35; ; i = v38 & v41 )
    {
      v34 = v85 + 16LL * i;
      v40 = *(const void **)v34;
      if ( *(_QWORD *)v34 == -1 )
        goto LABEL_66;
      if ( v40 == (const void *)-2LL )
      {
        v67 = (int *)((char *)s1a + 2) == 0;
      }
      else
      {
        if ( v28 != *(_QWORD *)(v34 + 8) )
          goto LABEL_44;
        v70 = v36;
        v73 = v37;
        v76 = i;
        v79 = v38;
        if ( !v28 )
          goto LABEL_69;
        v82 = v85 + 16LL * i;
        v66 = memcmp(s1a, v40, v28);
        v34 = v82;
        v38 = v79;
        i = v76;
        v37 = v73;
        v36 = v70;
        v67 = v66 == 0;
      }
      if ( v67 )
        goto LABEL_69;
      if ( v40 == (const void *)-2LL && !v37 )
        v37 = v34;
LABEL_44:
      v41 = v36 + i;
      ++v36;
    }
  }
  v89 = v32;
  v47 = sub_C94890(s1a, v28);
  v48 = 1;
  v34 = 0;
  v49 = v89 - 1;
  for ( j = (v89 - 1) & v47; ; j = v49 & v53 )
  {
    v51 = v84 + 16LL * j;
    v52 = *(const void **)v51;
    if ( *(_QWORD *)v51 == -1 )
    {
      v25 = (int *)((char *)s1a + 1) == 0;
    }
    else if ( v52 == (const void *)-2LL )
    {
      v25 = (int *)((char *)s1a + 2) == 0;
    }
    else
    {
      if ( v28 != *(_QWORD *)(v51 + 8) )
        goto LABEL_59;
      v72 = v84 + 16LL * j;
      v75 = v34;
      v78 = v48;
      v81 = j;
      v90 = v49;
      if ( !v28 )
        goto LABEL_24;
      v54 = memcmp(s1a, v52, v28);
      v49 = v90;
      j = v81;
      v48 = v78;
      v34 = v75;
      v51 = v72;
      v25 = v54 == 0;
    }
    if ( v25 )
      goto LABEL_24;
    if ( v52 == (const void *)-1LL )
      break;
LABEL_59:
    if ( v52 == (const void *)-2LL && !v34 )
      v34 = v51;
    v53 = v48 + j;
    ++v48;
  }
  v32 = v116;
  if ( !v34 )
    v34 = v51;
  ++s.m128i_i64[0];
  v55 = (_DWORD)v115 + 1;
  if ( 3 * (int)v116 <= (unsigned int)(4 * ((_DWORD)v115 + 1)) )
    goto LABEL_39;
  if ( (int)v116 - HIDWORD(v115) - v55 > (unsigned int)v116 >> 3 )
    goto LABEL_70;
  sub_BA8070((__int64)&s, v116);
  v34 = 0;
  v91 = v116;
  if ( !(_DWORD)v116 )
    goto LABEL_69;
  v86 = s.m128i_i64[1];
  v56 = sub_C94890(s1a, v28);
  v57 = 1;
  v37 = 0;
  v58 = v91 - 1;
  v59 = (v91 - 1) & v56;
  while ( 2 )
  {
    v34 = v86 + 16LL * v59;
    v60 = *(const void **)v34;
    if ( *(_QWORD *)v34 != -1 )
    {
      if ( v60 == (const void *)-2LL )
      {
        v69 = (int *)((char *)s1a + 2) == 0;
      }
      else
      {
        if ( v28 != *(_QWORD *)(v34 + 8) )
        {
LABEL_83:
          if ( v37 || v60 != (const void *)-2LL )
            v34 = v37;
          v61 = v57 + v59;
          v37 = v34;
          ++v57;
          v59 = v58 & v61;
          continue;
        }
        v71 = v57;
        v74 = v37;
        v77 = v59;
        v80 = v58;
        if ( !v28 )
          goto LABEL_69;
        v83 = v86 + 16LL * v59;
        v68 = memcmp(s1a, v60, v28);
        v34 = v83;
        v58 = v80;
        v59 = v77;
        v37 = v74;
        v57 = v71;
        v69 = v68 == 0;
      }
      if ( v69 )
        goto LABEL_69;
      if ( v60 == (const void *)-1LL )
        goto LABEL_67;
      goto LABEL_83;
    }
    break;
  }
LABEL_66:
  if ( s1a == (int *)-1LL )
    goto LABEL_69;
LABEL_67:
  if ( v37 )
    v34 = v37;
LABEL_69:
  v55 = (_DWORD)v115 + 1;
LABEL_70:
  LODWORD(v115) = v55;
  if ( *(_QWORD *)v34 != -1 )
    --HIDWORD(v115);
  *(_QWORD *)(v34 + 8) = v28;
  *(_QWORD *)v34 = s1a;
LABEL_24:
  if ( (unsigned int)sub_26E68F0(*(_QWORD **)(a1 + 8), (__int64)&s) )
  {
    sub_C7D6A0(s.m128i_i64[1], 16LL * (unsigned int)v116, 8);
    return 0;
  }
  v26 = 0;
  if ( *(_QWORD *)a3 )
    v26 = *(_QWORD *)(a3 + 8);
  v10 = sub_26C7880(*(_QWORD **)(a1 + 8), *(int **)a3, v26);
  sub_C7D6A0(s.m128i_i64[1], 16LL * (unsigned int)v116, 8);
  if ( !v10 )
    return 0;
LABEL_6:
  v11 = *(_QWORD *)(a2 + 80);
  for ( k = 0; v11 != a2 + 72; ++k )
    v11 = *(_QWORD *)(v11 + 8);
  if ( (unsigned int)dword_4FF8808 > k || (unsigned __int64)(unsigned int)dword_4FF8808 > *(_QWORD *)(v10 + 112) )
    return 0;
  v13 = byte_4F838D4[0];
  if ( !byte_4F838D4[0] )
    goto LABEL_15;
  v87 = *(_QWORD *)(a1 + 24);
  s.m128i_i64[0] = sub_B2D7E0(a2, "sample-profile-suffix-elision-policy", 0x24u);
  s1b = (void *)sub_A72240(s.m128i_i64);
  v95 = v14;
  *(_QWORD *)&v15 = sub_BD5D20(a2);
  v16 = sub_C16140(v15, (__int64)s1b, v95);
  v18 = sub_B2F650(v16, v17);
  v19 = *(unsigned int *)(v87 + 24);
  v20 = *(_QWORD *)(v87 + 8);
  if ( !(_DWORD)v19 )
  {
LABEL_15:
    v105 = &v103;
    v106 = &v103;
    v103 = 0;
    v104 = 0;
    v107 = 0;
    sub_26E1C80(a1, a2, (__int64)v102);
    v111 = &v109;
    v112 = &v109;
    v109 = 0;
    v110 = 0;
    v113 = 0;
    sub_26E39C0(a1, v10, &v108);
    v96 = 0;
    v13 = 0;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    sub_26E1700(a1, (__int64)v102, (__int64)&v108, (unsigned __int64 *)&v96, (unsigned __int64 *)&v99);
    v24 = v99;
    if ( (unsigned int)dword_4FF8728 <= 0xAAAAAAAAAAAAAAABLL * ((v97 - (__int64)v96) >> 3)
      && (v13 = 0, (unsigned int)dword_4FF8728 <= 0xAAAAAAAAAAAAAAABLL * ((v100 - (__int64)v99) >> 3)) )
    {
      sub_26E28C0(&s, a1, &v96, &v99, 0);
      if ( v116 < 0 )
        v42 = (float)(v116 & 1 | (unsigned int)((unsigned __int64)v116 >> 1))
            + (float)(v116 & 1 | (unsigned int)((unsigned __int64)v116 >> 1));
      else
        v42 = (float)(int)v116;
      v43 = 0xAAAAAAAAAAAAAAABLL * ((v100 - (__int64)v99) >> 3);
      if ( v43 < 0 )
      {
        v62 = v43 & 1 | ((0xAAAAAAAAAAAAAAABLL * ((v100 - (__int64)v99) >> 3)) >> 1);
        v44 = (float)(int)v62 + (float)(int)v62;
      }
      else
      {
        v44 = (float)(int)v43;
      }
      v45 = v115;
      LOBYTE(v13) = (float)((float)(v42 / v44) * 100.0) > (float)(int)qword_4FF88E8;
      while ( v45 )
      {
        v46 = (unsigned __int64)v45;
        v45 = (_QWORD *)*v45;
        j_j___libc_free_0(v46);
      }
      memset((void *)s.m128i_i64[0], 0, 8 * s.m128i_i64[1]);
      v116 = 0;
      v115 = 0;
      if ( (char *)s.m128i_i64[0] != &v117 )
        j_j___libc_free_0(s.m128i_u64[0]);
      v24 = v99;
      if ( !v99 )
        goto LABEL_19;
    }
    else if ( !v99 )
    {
LABEL_19:
      if ( v96 )
        j_j___libc_free_0((unsigned __int64)v96);
      sub_26E0760(v110);
      sub_26E0760(v104);
      return v13;
    }
    j_j___libc_free_0((unsigned __int64)v24);
    goto LABEL_19;
  }
  v21 = (v19 - 1) & (((0xBF58476D1CE4E5B9LL * v18) >> 31) ^ (484763065 * v18));
  v22 = (__int64 *)(v20 + 24LL * v21);
  v23 = *v22;
  if ( v18 != *v22 )
  {
    v63 = 1;
    while ( v23 != -1 )
    {
      v64 = v63 + 1;
      v65 = ((_DWORD)v19 - 1) & (v21 + v63);
      v21 = v65;
      v22 = (__int64 *)(v20 + 24 * v65);
      v23 = *v22;
      if ( v18 == *v22 )
        goto LABEL_13;
      v63 = v64;
    }
    goto LABEL_15;
  }
LABEL_13:
  if ( v22 == (__int64 *)(v20 + 24 * v19) || v22[2] != *(_QWORD *)(v10 + 8) )
    goto LABEL_15;
  return v13;
}
