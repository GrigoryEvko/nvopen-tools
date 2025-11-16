// Function: sub_ADB270
// Address: 0xadb270
//
__int64 __fastcall sub_ADB270(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8)
{
  __int64 v9; // r12
  __int64 v10; // rbx
  unsigned __int8 v11; // dl
  __int16 v12; // ax
  __int64 v13; // rcx
  __int64 v14; // rdx
  void *v15; // rax
  bool v16; // zf
  __int64 v17; // rax
  int v18; // eax
  __m128i v19; // xmm0
  __m128i v20; // xmm1
  int v21; // eax
  __int64 *v22; // rcx
  int v23; // r11d
  int v24; // r10d
  int i; // r8d
  __int64 *v26; // r9
  __int64 v27; // r13
  __int64 v28; // rax
  int v29; // r8d
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // r12
  __int64 v33; // rbx
  __int64 v34; // r13
  _QWORD *v35; // rdi
  __int64 v36; // rdi
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // r13
  int v41; // r13d
  unsigned int v42; // eax
  __int64 v43; // rsi
  int v44; // ecx
  __int64 v45; // r12
  unsigned int v47; // esi
  int v48; // eax
  __int64 *v49; // rdx
  int v50; // eax
  const void *v51; // rax
  void *v52; // rdx
  int v53; // eax
  __int64 v54; // rax
  int v55; // edi
  char v56; // al
  char v57; // al
  __int64 *v58; // [rsp+0h] [rbp-180h]
  __int64 *v59; // [rsp+8h] [rbp-178h]
  __int64 *v60; // [rsp+8h] [rbp-178h]
  __int64 *v61; // [rsp+8h] [rbp-178h]
  __int64 *v62; // [rsp+10h] [rbp-170h]
  int v63; // [rsp+10h] [rbp-170h]
  __int64 *v64; // [rsp+10h] [rbp-170h]
  int v65; // [rsp+18h] [rbp-168h]
  int v66; // [rsp+18h] [rbp-168h]
  int v67; // [rsp+18h] [rbp-168h]
  int v68; // [rsp+1Ch] [rbp-164h]
  int v69; // [rsp+1Ch] [rbp-164h]
  int v70; // [rsp+1Ch] [rbp-164h]
  int v71; // [rsp+20h] [rbp-160h]
  void *v72; // [rsp+20h] [rbp-160h]
  int v73; // [rsp+20h] [rbp-160h]
  __int64 v74; // [rsp+28h] [rbp-158h]
  __int64 v75; // [rsp+28h] [rbp-158h]
  __int64 *v78; // [rsp+48h] [rbp-138h] BYREF
  __int64 *v79; // [rsp+50h] [rbp-130h] BYREF
  int v80; // [rsp+58h] [rbp-128h]
  __int64 v81; // [rsp+60h] [rbp-120h] BYREF
  char v82; // [rsp+70h] [rbp-110h]
  __int64 v83; // [rsp+80h] [rbp-100h] BYREF
  __int16 v84; // [rsp+88h] [rbp-F8h] BYREF
  __m128i v85; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v86; // [rsp+A0h] [rbp-E0h] BYREF
  void *v87; // [rsp+B0h] [rbp-D0h] BYREF
  void *v88; // [rsp+B8h] [rbp-C8h] BYREF
  unsigned int v89; // [rsp+C0h] [rbp-C0h]
  void *v90; // [rsp+C8h] [rbp-B8h] BYREF
  unsigned int v91; // [rsp+D0h] [rbp-B0h]
  char v92; // [rsp+D8h] [rbp-A8h]
  unsigned __int64 v93; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v94; // [rsp+E8h] [rbp-98h] BYREF
  __m128i v95; // [rsp+F8h] [rbp-88h] BYREF
  void *s1[2]; // [rsp+108h] [rbp-78h] BYREF
  void *v97; // [rsp+118h] [rbp-68h]
  void *v98; // [rsp+120h] [rbp-60h] BYREF
  unsigned int v99; // [rsp+128h] [rbp-58h]
  void *v100; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v101; // [rsp+138h] [rbp-48h]
  char v102; // [rsp+140h] [rbp-40h]

  v9 = a1;
  v10 = a4;
  v11 = *(_BYTE *)(a4 + 1);
  v12 = *(_WORD *)(a4 + 2);
  v13 = 0;
  BYTE1(v93) = v11 >> 1;
  v14 = 0;
  LOBYTE(v93) = v12;
  v94.m128i_i64[0] = a2;
  v94.m128i_i64[1] = a3;
  if ( v12 == 63 )
  {
    v13 = sub_AC35F0(v10);
    v12 = *(_WORD *)(v10 + 2);
  }
  v95.m128i_i64[0] = v13;
  v95.m128i_i64[1] = v14;
  if ( v12 == 34 )
  {
    v15 = (void *)sub_AC5180(v10);
    v16 = *(_WORD *)(v10 + 2) == 34;
    s1[0] = v15;
    if ( v16 )
    {
      sub_AC51A0((__int64)&s1[1], v10);
      v17 = *(_QWORD *)(v10 + 8);
      v92 = 0;
      v83 = v17;
      v85 = _mm_loadu_si128(&v94);
      v84 = v93;
      v86 = _mm_loadu_si128(&v95);
      v87 = s1[0];
      if ( (_BYTE)v100 )
      {
        v92 = 1;
        v89 = (unsigned int)v97;
        v88 = s1[1];
        v91 = v99;
        v90 = v98;
      }
      goto LABEL_7;
    }
  }
  else
  {
    v15 = 0;
  }
  v30 = *(_QWORD *)(v10 + 8);
  v87 = v15;
  v92 = 0;
  v83 = v30;
  v85 = _mm_loadu_si128(&v94);
  v84 = v93;
  v86 = _mm_loadu_si128(&v95);
LABEL_7:
  v93 = sub_AC61D0((__int64 *)v86.m128i_i64[0], v86.m128i_i64[0] + 4 * v86.m128i_i64[1]);
  v79 = (__int64 *)sub_AC5F60((__int64 *)v85.m128i_i64[0], v85.m128i_i64[0] + 8 * v85.m128i_i64[1]);
  LODWORD(v93) = sub_AC5EC0((char *)&v84, (char *)&v84 + 1, (__int64 *)&v79, (__int64 *)&v93, (__int64 *)&v87);
  v18 = sub_AC7AE0(&v83, &v93);
  v102 = 0;
  LODWORD(v93) = v18;
  v19 = _mm_loadu_si128(&v85);
  v20 = _mm_loadu_si128(&v86);
  v94.m128i_i64[0] = v83;
  v95 = v19;
  v94.m128i_i16[4] = v84;
  *(__m128i *)s1 = v20;
  v97 = v87;
  if ( v92 )
  {
    v99 = v89;
    if ( v89 > 0x40 )
      sub_C43780(&v98, &v88);
    else
      v98 = v88;
    v101 = v91;
    if ( v91 > 0x40 )
      sub_C43780(&v100, &v90);
    else
      v100 = v90;
    v21 = *(_DWORD *)(a1 + 24);
    v22 = *(__int64 **)(a1 + 8);
    v102 = 1;
    if ( !v21 )
      goto LABEL_21;
  }
  else
  {
    v21 = *(_DWORD *)(a1 + 24);
    v22 = *(__int64 **)(a1 + 8);
    if ( !v21 )
      goto LABEL_21;
  }
  v23 = v21 - 1;
  v24 = 1;
  for ( i = (v21 - 1) & v93; ; i = v23 & v29 )
  {
    v26 = &v22[i];
    v27 = *v26;
    v28 = *v26;
    if ( *v26 == -4096 )
    {
LABEL_40:
      v39 = *(_QWORD *)(v9 + 8);
      v40 = *(unsigned int *)(v9 + 24);
      v22 = (__int64 *)(v39 + 8 * v40);
      goto LABEL_41;
    }
    if ( v27 == -8192 )
      goto LABEL_39;
    if ( v94.m128i_i64[0] != *(_QWORD *)(v27 + 8) )
      goto LABEL_13;
    if ( v94.m128i_u8[8] != *(unsigned __int16 *)(v27 + 2) )
      goto LABEL_13;
    if ( v94.m128i_i8[9] != *(_BYTE *)(v27 + 1) >> 1 )
      goto LABEL_13;
    v36 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
    if ( v95.m128i_i64[1] != v36 )
      goto LABEL_13;
    if ( (*(_DWORD *)(v27 + 4) & 0x7FFFFFF) != 0 )
    {
      v37 = (_QWORD *)v95.m128i_i64[0];
      v38 = (_QWORD *)(v27 - 32 * v36);
      while ( *v37 == *v38 )
      {
        ++v37;
        v38 += 4;
        if ( (_QWORD *)(v95.m128i_i64[0] + 8LL * ((*(_DWORD *)(v27 + 4) & 0x7FFFFFFu) - 1) + 8) == v37 )
          goto LABEL_36;
      }
      goto LABEL_13;
    }
LABEL_36:
    if ( *(_WORD *)(v27 + 2) == 63 )
    {
      v59 = &v22[i];
      v62 = v22;
      v65 = v24;
      v68 = i;
      v71 = v23;
      v51 = (const void *)sub_AC35F0(v27);
      v23 = v71;
      i = v68;
      v24 = v65;
      v22 = v62;
      v26 = v59;
      if ( s1[1] != v52 )
      {
        v27 = *v59;
LABEL_38:
        v28 = v27;
        goto LABEL_39;
      }
      if ( 4 * (__int64)s1[1] )
      {
        v53 = memcmp(s1[0], v51, 4 * (__int64)s1[1]);
        v23 = v71;
        i = v68;
        v24 = v65;
        v22 = v62;
        v26 = v59;
        if ( v53 )
          goto LABEL_77;
      }
    }
    else if ( s1[1] )
    {
      goto LABEL_38;
    }
    if ( *(_WORD *)(v27 + 2) != 34 )
    {
      if ( !v97 )
      {
LABEL_81:
        v82 = 0;
        if ( !v102 )
          goto LABEL_82;
      }
LABEL_77:
      v28 = *v26;
      goto LABEL_39;
    }
    v58 = v26;
    v60 = v22;
    v63 = v24;
    v66 = i;
    v69 = v23;
    v72 = v97;
    v54 = sub_AC5180(v27);
    v23 = v69;
    i = v66;
    v24 = v63;
    v22 = v60;
    v26 = v58;
    if ( v72 != (void *)v54 )
      goto LABEL_77;
    if ( *(_WORD *)(v27 + 2) != 34 )
      goto LABEL_81;
    sub_AC51A0((__int64)&v79, v27);
    v23 = v69;
    i = v66;
    v24 = v63;
    v22 = v60;
    v26 = v58;
    if ( !v102 )
    {
      if ( !v82 )
        goto LABEL_82;
      sub_9963D0((__int64)&v79);
      v23 = v69;
      i = v66;
      v24 = v63;
      v22 = v60;
      v26 = v58;
      goto LABEL_77;
    }
    if ( !v82 )
      goto LABEL_77;
    if ( v99 == v80 )
    {
      v56 = sub_AAD8B0((__int64)&v98, &v79);
      v23 = v69;
      i = v66;
      v24 = v63;
      v22 = v60;
      v26 = v58;
      if ( v56 )
      {
        v57 = sub_AAD8B0((__int64)&v100, &v81);
        v23 = v69;
        i = v66;
        v24 = v63;
        v22 = v60;
        v26 = v58;
        if ( v57 )
          break;
      }
    }
    v61 = v26;
    v64 = v22;
    v67 = v24;
    v70 = i;
    v73 = v23;
    sub_9963D0((__int64)&v79);
    v23 = v73;
    i = v70;
    v24 = v67;
    v28 = *v61;
    v22 = v64;
LABEL_39:
    if ( v28 == -4096 )
      goto LABEL_40;
LABEL_13:
    v29 = v24 + i;
    ++v24;
  }
  sub_9963D0((__int64)&v79);
  v26 = v58;
LABEL_82:
  v39 = *(_QWORD *)(v9 + 8);
  v22 = (__int64 *)(v39 + 8LL * *(unsigned int *)(v9 + 24));
  LODWORD(v40) = *(_DWORD *)(v9 + 24);
  if ( v26 != v22 )
  {
    v45 = *v26;
    goto LABEL_51;
  }
LABEL_41:
  v75 = v39;
  if ( (_DWORD)v40 )
  {
    v41 = v40 - 1;
    v42 = v41 & sub_ACF990(v10);
    v22 = (__int64 *)(v75 + 8LL * v42);
    v43 = *v22;
    if ( *v22 != v10 )
    {
      v44 = 1;
      while ( v43 != -4096 )
      {
        v55 = v44 + 1;
        v42 = v41 & (v44 + v42);
        v22 = (__int64 *)(v75 + 8LL * v42);
        v43 = *v22;
        if ( *v22 == v10 )
          goto LABEL_21;
        v44 = v55;
      }
      v22 = (__int64 *)(*(_QWORD *)(v9 + 8) + 8LL * *(unsigned int *)(v9 + 24));
    }
  }
LABEL_21:
  *v22 = -8192;
  --*(_DWORD *)(v9 + 16);
  ++*(_DWORD *)(v9 + 20);
  if ( a7 == 1 )
  {
    sub_AC2B30(v10 + 32 * (a8 - (unsigned __int64)(*(_DWORD *)(v10 + 4) & 0x7FFFFFF)), a6);
  }
  else
  {
    v31 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v31 )
    {
      v74 = v9;
      v32 = v10;
      v33 = 0;
      v34 = v31 - 1;
      while ( 1 )
      {
        v35 = (_QWORD *)(v32 + 32 * (v33 - v31));
        if ( a5 == *v35 )
          sub_AC2B30((__int64)v35, a6);
        if ( v33 == v34 )
          break;
        ++v33;
        v31 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
      }
      v10 = v32;
      v9 = v74;
    }
  }
  if ( !(unsigned __int8)sub_AC8350(v9, (__int64)&v93, &v78) )
  {
    v47 = *(_DWORD *)(v9 + 24);
    v48 = *(_DWORD *)(v9 + 16);
    v49 = v78;
    ++*(_QWORD *)v9;
    v50 = v48 + 1;
    v79 = v49;
    if ( 4 * v50 >= 3 * v47 )
    {
      v47 *= 2;
    }
    else if ( v47 - *(_DWORD *)(v9 + 20) - v50 > v47 >> 3 )
    {
LABEL_68:
      *(_DWORD *)(v9 + 16) = v50;
      if ( *v49 != -4096 )
        --*(_DWORD *)(v9 + 20);
      *v49 = v10;
      goto LABEL_50;
    }
    sub_AD4030(v9, v47);
    sub_AC8350(v9, (__int64)&v93, &v79);
    v49 = v79;
    v50 = *(_DWORD *)(v9 + 16) + 1;
    goto LABEL_68;
  }
LABEL_50:
  v45 = 0;
LABEL_51:
  if ( v102 )
  {
    v102 = 0;
    if ( v101 > 0x40 && v100 )
      j_j___libc_free_0_0(v100);
    if ( v99 > 0x40 && v98 )
      j_j___libc_free_0_0(v98);
  }
  if ( v92 )
  {
    v92 = 0;
    if ( v91 > 0x40 && v90 )
      j_j___libc_free_0_0(v90);
    if ( v89 > 0x40 && v88 )
      j_j___libc_free_0_0(v88);
  }
  return v45;
}
