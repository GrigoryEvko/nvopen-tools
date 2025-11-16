// Function: sub_37CE0D0
// Address: 0x37ce0d0
//
void __fastcall sub_37CE0D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 v6; // r15
  __int64 v7; // rbx
  bool v8; // zf
  __int64 v9; // rax
  __int16 v10; // si
  bool v11; // al
  __int64 v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // r15
  const __m128i *v15; // rbx
  const __m128i *v16; // rcx
  const __m128i *v17; // rax
  signed __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // r8
  char **v21; // r9
  __int64 v22; // rdx
  const __m128i *v23; // r12
  unsigned __int64 v24; // rsi
  __m128i *i; // rcx
  __int32 v26; // edi
  const __m128i *v27; // r15
  __m128i *v28; // rax
  unsigned __int64 v29; // r11
  __m128i *v30; // rax
  const __m128i *v31; // r15
  __int64 v32; // rdi
  __m128i *v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // rbx
  __int64 v36; // r13
  int *v37; // rax
  int *v38; // r15
  char *v39; // rdi
  __int64 v40; // rax
  char *v41; // rdx
  int v42; // esi
  unsigned __int64 v43; // rdi
  int v44; // eax
  __int64 v45; // rdi
  int v46; // eax
  unsigned int v47; // ecx
  int *v48; // rsi
  int v49; // r8d
  unsigned __int64 v50; // r11
  int *v51; // r14
  __int64 v52; // r12
  __int64 v53; // rax
  bool v54; // al
  __int64 v55; // rsi
  __int64 v56; // rcx
  int *v57; // rdi
  int *v58; // rax
  char *v59; // r15
  char *v60; // r15
  int v61; // esi
  int v62; // r9d
  __int64 v63; // [rsp+8h] [rbp-158h]
  int v64; // [rsp+18h] [rbp-148h]
  __int64 v65; // [rsp+18h] [rbp-148h]
  int *v66; // [rsp+20h] [rbp-140h]
  __int64 v67; // [rsp+28h] [rbp-138h]
  char **v68; // [rsp+28h] [rbp-138h]
  char **v69; // [rsp+28h] [rbp-138h]
  char **v70; // [rsp+28h] [rbp-138h]
  char **v71; // [rsp+28h] [rbp-138h]
  unsigned int v72; // [rsp+30h] [rbp-130h]
  __int64 v73; // [rsp+30h] [rbp-130h]
  __int64 v74; // [rsp+38h] [rbp-128h]
  _BYTE *v75; // [rsp+38h] [rbp-128h]
  unsigned int v76; // [rsp+4Ch] [rbp-114h] BYREF
  __int64 v77; // [rsp+50h] [rbp-110h] BYREF
  bool v78; // [rsp+58h] [rbp-108h]
  bool v79; // [rsp+59h] [rbp-107h]
  __int64 v80; // [rsp+60h] [rbp-100h] BYREF
  int v81; // [rsp+68h] [rbp-F8h] BYREF
  unsigned __int16 v82; // [rsp+70h] [rbp-F0h]
  char v83; // [rsp+78h] [rbp-E8h]
  __int64 v84; // [rsp+80h] [rbp-E0h]
  __m128i v85; // [rsp+90h] [rbp-D0h] BYREF
  __m128i v86; // [rsp+A0h] [rbp-C0h]
  __int64 v87; // [rsp+B0h] [rbp-B0h]
  __m128i v88; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i v89; // [rsp+D0h] [rbp-90h]
  __int64 v90; // [rsp+E0h] [rbp-80h]
  char v91; // [rsp+E8h] [rbp-78h]
  unsigned __int64 v92; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v93; // [rsp+F8h] [rbp-68h]
  _BYTE v94[16]; // [rsp+100h] [rbp-60h] BYREF
  _BYTE *v95; // [rsp+110h] [rbp-50h]

  v2 = a1;
  v4 = sub_B10CD0(a2 + 56);
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
  {
    if ( *(_DWORD *)(v4 - 24) != 2 )
    {
LABEL_3:
      v6 = 0;
      goto LABEL_4;
    }
    v34 = *(_QWORD *)(v4 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_3;
    v34 = v4 - 16 - 8LL * ((v5 >> 2) & 0xF);
  }
  v6 = *(_QWORD *)(v34 + 8);
LABEL_4:
  v7 = sub_2E891C0(a2);
  v80 = sub_2E89170(a2);
  if ( v7 )
    sub_AF47B0((__int64)&v81, *(unsigned __int64 **)(v7 + 16), *(unsigned __int64 **)(v7 + 24));
  else
    v83 = 0;
  v8 = *(_WORD *)(a2 + 68) == 15;
  v84 = v6;
  v79 = v8;
  v9 = sub_2E891C0(a2);
  v10 = *(_WORD *)(a2 + 68);
  v77 = v9;
  v11 = 0;
  if ( v10 == 14 )
    v11 = *(_BYTE *)(*(_QWORD *)(a2 + 32) + 40LL) == 1;
  v12 = *(_QWORD *)(a1 + 32);
  v78 = v11;
  v13 = *(unsigned int *)(v12 + 24);
  v74 = *(_QWORD *)(v12 + 8);
  if ( (_DWORD)v13 )
  {
    v92 = 0;
    v94[8] = 0;
    v95 = 0;
    v76 = 0;
    if ( v83 )
      v76 = v82 | (v81 << 16);
    v88.m128i_i64[0] = v6;
    v67 = v12;
    v85.m128i_i64[0] = v80;
    v64 = 1;
    v72 = (v13 - 1) & sub_F11290(v85.m128i_i64, &v76, v88.m128i_i64);
    while ( 1 )
    {
      v14 = v74 + 48LL * v72;
      if ( sub_F34140((__int64)&v80, v14) )
        break;
      if ( sub_F34140(v14, (__int64)&v92) )
      {
        v13 = *(unsigned int *)(v67 + 24);
        v10 = *(_WORD *)(a2 + 68);
        v74 = *(_QWORD *)(v67 + 8);
        goto LABEL_84;
      }
      v72 = (v13 - 1) & (v64 + v72);
      ++v64;
    }
    v10 = *(_WORD *)(a2 + 68);
    if ( v14 )
      goto LABEL_14;
    v13 = *(unsigned int *)(v67 + 24);
    v74 = *(_QWORD *)(v67 + 8);
  }
LABEL_84:
  v14 = 48 * v13 + v74;
LABEL_14:
  v15 = *(const __m128i **)(a2 + 32);
  v76 = *(_DWORD *)(v14 + 40);
  if ( v10 == 14 )
  {
    v16 = (const __m128i *)((char *)v15 + 40);
    v17 = v15;
    goto LABEL_74;
  }
  v16 = (const __m128i *)((char *)v15 + 40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
  v17 = v15 + 5;
  if ( v10 == 15 )
  {
    if ( v17 != v16 )
    {
LABEL_74:
      while ( v17->m128i_i8[0] || v17->m128i_i32[2] )
      {
        v17 = (const __m128i *)((char *)v17 + 40);
        if ( v16 == v17 )
        {
          v16 = (const __m128i *)((char *)v15 + 40);
          v17 = v15;
          if ( v10 == 14 )
            goto LABEL_78;
          v16 = (const __m128i *)((char *)v15 + 40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
          v17 = v15 + 5;
          goto LABEL_16;
        }
      }
    }
    goto LABEL_40;
  }
LABEL_16:
  v18 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v16 - (char *)v17) >> 3);
  v19 = v18 >> 2;
  if ( v18 >> 2 > 0 )
  {
    while ( v17->m128i_i8[0] )
    {
      if ( !v17[2].m128i_i8[8] )
      {
        v17 = (const __m128i *)((char *)v17 + 40);
        goto LABEL_23;
      }
      if ( !v17[5].m128i_i8[0] )
      {
        v17 += 5;
        goto LABEL_23;
      }
      if ( !v17[7].m128i_i8[8] )
      {
        v17 = (const __m128i *)((char *)v17 + 120);
        goto LABEL_23;
      }
      v17 += 10;
      if ( !--v19 )
      {
        v18 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v16 - (char *)v17) >> 3);
        goto LABEL_113;
      }
    }
    goto LABEL_23;
  }
LABEL_113:
  if ( v18 != 2 )
  {
    if ( v18 != 3 )
    {
      if ( v18 != 1 )
        goto LABEL_40;
      goto LABEL_78;
    }
    if ( !v17->m128i_i8[0] )
      goto LABEL_23;
    v17 = (const __m128i *)((char *)v17 + 40);
  }
  if ( !v17->m128i_i8[0] )
    goto LABEL_23;
  v17 = (const __m128i *)((char *)v17 + 40);
LABEL_78:
  if ( !v17->m128i_i8[0] )
  {
LABEL_23:
    if ( v17 != v16 )
    {
      v20 = (__int64)&v15[2].m128i_i64[1];
      v92 = (unsigned __int64)v94;
      v93 = 0x100000000LL;
      if ( v10 != 14 )
      {
        v20 = (__int64)&v15->m128i_i64[5 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
        v15 += 5;
      }
      v21 = (char **)&v92;
      if ( (const __m128i *)v20 != v15 )
      {
        v22 = 0;
        v21 = (char **)&v92;
        v23 = (const __m128i *)v20;
        v24 = 1;
        for ( i = (__m128i *)v94; ; i = (__m128i *)v92 )
        {
          v29 = v22 + 1;
          v30 = &i[3 * v22];
          if ( v15->m128i_i8[0] )
          {
            v31 = &v88;
            v85 = _mm_loadu_si128(v15);
            v86 = _mm_loadu_si128(v15 + 1);
            v32 = v15[2].m128i_i64[0];
            v91 = 1;
            v87 = v32;
            v90 = v32;
            v88 = v85;
            v89 = v86;
            if ( v24 < v29 )
            {
              if ( i > &v88 || v30 <= &v88 )
              {
                v31 = &v88;
                v70 = v21;
                sub_C8D5F0((__int64)v21, v94, v22 + 1, 0x30u, v20, (__int64)v21);
                i = (__m128i *)v92;
                v22 = (unsigned int)v93;
                v21 = v70;
              }
              else
              {
                v60 = (char *)((char *)&v88 - (char *)i);
                v69 = v21;
                sub_C8D5F0((__int64)v21, v94, v22 + 1, 0x30u, v20, (__int64)v21);
                i = (__m128i *)v92;
                v22 = (unsigned int)v93;
                v21 = v69;
                v31 = (const __m128i *)&v60[v92];
              }
            }
            v15 = (const __m128i *)((char *)v15 + 40);
            v33 = &i[3 * v22];
            *v33 = _mm_loadu_si128(v31);
            v33[1] = _mm_loadu_si128(v31 + 1);
            v33[2] = _mm_loadu_si128(v31 + 2);
            LODWORD(v93) = v93 + 1;
            if ( v23 == v15 )
              break;
          }
          else
          {
            v26 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 16) + 64LL) + 4LL * v15->m128i_u32[2]);
            v91 = 0;
            v27 = &v88;
            v88.m128i_i32[0] = v26;
            if ( v24 < v29 )
            {
              if ( i > &v88 || v30 <= &v88 )
              {
                v27 = &v88;
                v71 = v21;
                sub_C8D5F0((__int64)v21, v94, v22 + 1, 0x30u, v20, (__int64)v21);
                i = (__m128i *)v92;
                v22 = (unsigned int)v93;
                v21 = v71;
              }
              else
              {
                v59 = (char *)((char *)&v88 - (char *)i);
                v68 = v21;
                sub_C8D5F0((__int64)v21, v94, v22 + 1, 0x30u, v20, (__int64)v21);
                i = (__m128i *)v92;
                v22 = (unsigned int)v93;
                v21 = v68;
                v27 = (const __m128i *)&v59[v92];
              }
            }
            v15 = (const __m128i *)((char *)v15 + 40);
            v28 = &i[3 * v22];
            *v28 = _mm_loadu_si128(v27);
            v28[1] = _mm_loadu_si128(v27 + 1);
            v28[2] = _mm_loadu_si128(v27 + 2);
            LODWORD(v93) = v93 + 1;
            if ( v23 == v15 )
              break;
          }
          v22 = (unsigned int)v93;
          v24 = HIDWORD(v93);
        }
      }
      sub_37CD070(v2, a2, (__int64)&v77, v21);
      if ( (_BYTE *)v92 != v94 )
        _libc_free(v92);
      return;
    }
  }
LABEL_40:
  sub_37BC1E0((__int64 **)&v88, (__int64 *)(v2 + 3440), (int *)&v76);
  v65 = v89.m128i_i64[0];
  if ( v89.m128i_i64[0] == *(_QWORD *)(v2 + 3448) + 88LL * *(unsigned int *)(v2 + 3464) )
    goto LABEL_59;
  sub_37B9C30((__int64)&v92, (__int64 *)(v89.m128i_i64[0] + 8));
  v35 = v92;
  v36 = v93;
  v75 = v95;
  if ( v95 == (_BYTE *)v92 )
    goto LABEL_56;
  v63 = v2;
  v73 = v2 + 3408;
  do
  {
    v85.m128i_i32[0] = *(_DWORD *)v35;
    v37 = sub_37BEF10(v73, v85.m128i_i32);
    v38 = v37;
    if ( *((_QWORD *)v37 + 9) )
    {
      v50 = *((_QWORD *)v37 + 6);
      v51 = v37 + 10;
      v66 = v37 + 10;
      if ( v50 )
      {
        v52 = *((_QWORD *)v37 + 6);
        while ( 1 )
        {
          while ( *(_DWORD *)(v52 + 32) < v76 )
          {
            v52 = *(_QWORD *)(v52 + 24);
            if ( !v52 )
              goto LABEL_68;
          }
          v53 = *(_QWORD *)(v52 + 16);
          if ( *(_DWORD *)(v52 + 32) <= v76 )
            break;
          v51 = (int *)v52;
          v52 = *(_QWORD *)(v52 + 16);
          if ( !v53 )
          {
LABEL_68:
            v54 = v51 == v66;
            goto LABEL_69;
          }
        }
        v55 = *(_QWORD *)(v52 + 24);
        while ( v55 )
        {
          if ( v76 >= *(_DWORD *)(v55 + 32) )
          {
            v55 = *(_QWORD *)(v55 + 24);
          }
          else
          {
            v51 = (int *)v55;
            v55 = *(_QWORD *)(v55 + 16);
          }
        }
        while ( v53 )
        {
          while ( 1 )
          {
            v56 = *(_QWORD *)(v53 + 24);
            if ( v76 <= *(_DWORD *)(v53 + 32) )
              break;
            v53 = *(_QWORD *)(v53 + 24);
            if ( !v56 )
              goto LABEL_89;
          }
          v52 = v53;
          v53 = *(_QWORD *)(v53 + 16);
        }
LABEL_89:
        if ( *((_QWORD *)v38 + 7) != v52 || v66 != v51 )
        {
          for ( ; (int *)v52 != v51; --*((_QWORD *)v38 + 9) )
          {
            v57 = (int *)v52;
            v52 = sub_220EF30(v52);
            v58 = sub_220F330(v57, v66);
            j_j___libc_free_0((unsigned __int64)v58);
          }
          goto LABEL_53;
        }
      }
      else
      {
        v51 = v37 + 10;
        v54 = 1;
LABEL_69:
        if ( *((int **)v38 + 7) != v51 || !v54 )
          goto LABEL_53;
      }
      sub_37B80B0(v50);
      *((_QWORD *)v38 + 6) = 0;
      *((_QWORD *)v38 + 9) = 0;
      *((_QWORD *)v38 + 7) = v66;
      *((_QWORD *)v38 + 8) = v66;
      goto LABEL_53;
    }
    v39 = *(char **)v37;
    v40 = (unsigned int)v37[2];
    v41 = &v39[4 * v40];
    v42 = v40;
    if ( v39 == v41 )
      goto LABEL_53;
    while ( *(_DWORD *)v39 != v76 )
    {
      v39 += 4;
      if ( v41 == v39 )
        goto LABEL_53;
    }
    if ( v41 == v39 )
      goto LABEL_53;
    if ( v41 != v39 + 4 )
    {
      memmove(v39, v39 + 4, v41 - (v39 + 4));
      v42 = v38[2];
    }
    v35 += 48LL;
    for ( v38[2] = v42 - 1; v35 != v36; v35 += 48LL )
    {
      if ( !*(_BYTE *)(v35 + 40) )
        break;
LABEL_53:
      ;
    }
  }
  while ( v75 != (_BYTE *)v35 );
  v2 = v63;
LABEL_56:
  v43 = *(_QWORD *)(v65 + 8);
  if ( v43 != v65 + 24 )
    _libc_free(v43);
  *(_DWORD *)v65 = -2;
  --*(_DWORD *)(v2 + 3456);
  ++*(_DWORD *)(v2 + 3460);
LABEL_59:
  v44 = *(_DWORD *)(v2 + 3608);
  v45 = *(_QWORD *)(v2 + 3592);
  if ( v44 )
  {
    v46 = v44 - 1;
    v47 = v46 & (37 * v76);
    v48 = (int *)(v45 + 4LL * v47);
    v49 = *v48;
    if ( *v48 == v76 )
    {
LABEL_61:
      *v48 = -2;
      --*(_DWORD *)(v2 + 3600);
      ++*(_DWORD *)(v2 + 3604);
    }
    else
    {
      v61 = 1;
      while ( v49 != -1 )
      {
        v62 = v61 + 1;
        v47 = v46 & (v61 + v47);
        v48 = (int *)(v45 + 4LL * v47);
        v49 = *v48;
        if ( v76 == *v48 )
          goto LABEL_61;
        v61 = v62;
      }
    }
  }
}
