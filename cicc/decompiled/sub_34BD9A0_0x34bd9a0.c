// Function: sub_34BD9A0
// Address: 0x34bd9a0
//
__int64 __fastcall sub_34BD9A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // r13d
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rbx
  _BYTE *v9; // r13
  char v10; // r15
  int v11; // r11d
  int v12; // r14d
  __int64 v13; // rax
  unsigned int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v28; // rsi
  unsigned int m; // eax
  unsigned int *v30; // rdi
  unsigned int v31; // eax
  unsigned int v32; // eax
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r14
  const char *v37; // rax
  size_t v38; // rdx
  const __m128i *v39; // rbx
  const __m128i *v40; // r15
  int v41; // r13d
  __int64 v42; // rdi
  unsigned int j; // eax
  unsigned int v44; // eax
  __int64 v45; // rdi
  __int64 (*v46)(); // rax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 (*v49)(); // rax
  char v50; // al
  int v51; // r10d
  unsigned int i; // eax
  int v53; // r11d
  unsigned int v54; // eax
  int v55; // edx
  int v56; // r10d
  unsigned int k; // eax
  int v58; // r11d
  unsigned int v59; // eax
  int v60; // [rsp+4h] [rbp-BCh]
  int v61; // [rsp+4h] [rbp-BCh]
  __int64 v63; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v64; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v65; // [rsp+28h] [rbp-98h]
  __int64 v66; // [rsp+30h] [rbp-90h]
  unsigned int v67; // [rsp+38h] [rbp-88h]
  __int64 *v68; // [rsp+40h] [rbp-80h] BYREF
  const __m128i *v69; // [rsp+48h] [rbp-78h]
  __int64 *v70; // [rsp+50h] [rbp-70h]
  _BYTE v71[104]; // [rsp+58h] [rbp-68h] BYREF

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_DWORD *)(v3 + 880);
  if ( v4 == 3 )
  {
    v14 = 0;
    goto LABEL_20;
  }
  if ( v4 == 1 )
  {
    if ( (unsigned __int8)sub_34BC740((__int64 *)a2) )
    {
      v3 = *(_QWORD *)(a2 + 8);
      v14 = 0;
      goto LABEL_20;
    }
    sub_2E7A760(a2, 0);
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v33 = *(__int64 **)(a1 + 8);
    v64 = 0;
    v34 = *v33;
    v35 = v33[1];
    if ( v34 == v35 )
LABEL_116:
      BUG();
    while ( *(_UNKNOWN **)v34 != &unk_501695C )
    {
      v34 += 16;
      if ( v35 == v34 )
        goto LABEL_116;
    }
    v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(
            *(_QWORD *)(v34 + 8),
            &unk_501695C);
    v37 = sub_2E791E0((__int64 *)a2);
    sub_2D51470((__int64)&v68, v36, v37, v38);
    v14 = (unsigned __int8)v68;
    if ( !(_BYTE)v68 )
    {
      if ( v69 != (const __m128i *)v71 )
        _libc_free((unsigned __int64)v69);
      goto LABEL_19;
    }
    v39 = v69;
    v40 = &v69[(unsigned int)v70];
    if ( v69 != v40 )
    {
LABEL_47:
      if ( !v67 )
      {
        ++v64;
LABEL_77:
        sub_34BD700((__int64)&v64, 2 * v67);
        if ( v67 )
        {
          v5 = v39->m128i_u32[1];
          v51 = 1;
          v6 = v67 - 1;
          v7 = 0;
          for ( i = v6
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v39->m128i_i32[0]) << 32))) >> 31)
                   ^ (756364221 * v5)); ; i = v6 & v54 )
          {
            v42 = v65 + 24LL * i;
            v53 = *(_DWORD *)v42;
            if ( v39->m128i_i64[0] == *(_QWORD *)v42 )
              break;
            if ( v53 == -1 )
            {
              if ( *(_DWORD *)(v42 + 4) == -1 )
              {
LABEL_108:
                if ( v7 )
                  v42 = v7;
                v55 = v66 + 1;
                goto LABEL_91;
              }
            }
            else if ( v53 == -2 && *(_DWORD *)(v42 + 4) == -2 && !v7 )
            {
              v7 = v65 + 24LL * i;
            }
            v54 = v51 + i;
            ++v51;
          }
          goto LABEL_105;
        }
LABEL_114:
        LODWORD(v66) = v66 + 1;
        BUG();
      }
      v5 = v39->m128i_u32[1];
      v41 = 1;
      v42 = 0;
      for ( j = (v67 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v39->m128i_i32[0]) << 32))) >> 31)
               ^ (756364221 * v5)); ; j = (v67 - 1) & v44 )
      {
        v6 = v65 + 24LL * j;
        v7 = *(unsigned int *)v6;
        if ( v39->m128i_i64[0] == *(_QWORD *)v6 )
          goto LABEL_68;
        if ( (_DWORD)v7 == -1 )
        {
          if ( *(_DWORD *)(v6 + 4) == -1 )
          {
            if ( !v42 )
              v42 = v65 + 24LL * j;
            ++v64;
            v55 = v66 + 1;
            if ( 4 * ((int)v66 + 1) >= 3 * v67 )
              goto LABEL_77;
            v5 = v67 >> 3;
            if ( v67 - HIDWORD(v66) - v55 > (unsigned int)v5 )
              goto LABEL_91;
            sub_34BD700((__int64)&v64, v67);
            if ( !v67 )
              goto LABEL_114;
            v5 = v39->m128i_u32[1];
            v56 = 1;
            v6 = v67 - 1;
            v7 = 0;
            for ( k = v6
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v39->m128i_i32[0]) << 32))) >> 31)
                     ^ (756364221 * v5)); ; k = v6 & v59 )
            {
              v42 = v65 + 24LL * k;
              v58 = *(_DWORD *)v42;
              if ( v39->m128i_i64[0] == *(_QWORD *)v42 )
                break;
              if ( v58 == -1 )
              {
                if ( *(_DWORD *)(v42 + 4) == -1 )
                  goto LABEL_108;
              }
              else if ( v58 == -2 && *(_DWORD *)(v42 + 4) == -2 && !v7 )
              {
                v7 = v65 + 24LL * k;
              }
              v59 = v56 + k;
              ++v56;
            }
LABEL_105:
            v55 = v66 + 1;
LABEL_91:
            LODWORD(v66) = v55;
            if ( *(_DWORD *)v42 != -1 || *(_DWORD *)(v42 + 4) != -1 )
              --HIDWORD(v66);
            *(_QWORD *)v42 = v39->m128i_i64[0];
            *(__m128i *)(v42 + 8) = _mm_loadu_si128(v39);
LABEL_68:
            if ( v40 == ++v39 )
            {
              v4 = 1;
              v40 = v69;
              break;
            }
            goto LABEL_47;
          }
        }
        else if ( (_DWORD)v7 == -2 && *(_DWORD *)(v6 + 4) == -2 && !v42 )
        {
          v42 = v65 + 24LL * j;
        }
        v44 = v41 + j;
        ++v41;
      }
    }
    if ( v40 != (const __m128i *)v71 )
      _libc_free((unsigned __int64)v40);
  }
  else
  {
    sub_2E7A760(a2, 0);
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
  }
  v8 = *(_QWORD *)(a2 + 328);
  *(_DWORD *)(a2 + 588) = v4;
  v9 = (_BYTE *)(a2 + 320);
  if ( v8 == a2 + 320 )
    goto LABEL_18;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  do
  {
    while ( 1 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(a2 + 8) + 880LL) && (_DWORD)v66 )
      {
        v5 = *(unsigned int *)(v8 + 240);
        v28 = *(unsigned int *)(v8 + 244);
        v6 = v65;
        if ( !v67 )
          goto LABEL_57;
        v60 = 1;
        for ( m = (v67 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v28) | ((unsigned __int64)(unsigned int)(37 * v5) << 32))) >> 31)
                 ^ (756364221 * v28)); ; m = (v67 - 1) & v31 )
        {
          v30 = (unsigned int *)(v65 + 24LL * m);
          v7 = *v30;
          if ( __PAIR64__(v28, v5) == *(_QWORD *)v30 )
            break;
          if ( (_DWORD)v7 == -1 && v30[1] == -1 )
            goto LABEL_57;
          v31 = v60 + m;
          ++v60;
        }
        if ( v30 == (unsigned int *)(v65 + 24LL * v67) )
        {
LABEL_57:
          v61 = v11;
          v45 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 16LL);
          v46 = *(__int64 (**)())(*(_QWORD *)v45 + 128LL);
          if ( v46 == sub_2DAC790 )
            BUG();
          v47 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v46)(v45, v28, v67, v5, v65);
          v11 = v61;
          v48 = v47;
          v49 = *(__int64 (**)())(*(_QWORD *)v47 + 1456LL);
          if ( v49 == sub_2FDC7E0 || (v50 = ((__int64 (__fastcall *)(__int64, __int64))v49)(v48, v8), v11 = v61, v50) )
            *(_QWORD *)(v8 + 252) = unk_501EB38;
        }
        else
        {
          v32 = v30[4];
          *(_DWORD *)(v8 + 252) = 0;
          *(_DWORD *)(v8 + 256) = v32;
        }
      }
      else
      {
        *(_DWORD *)(v8 + 252) = 0;
        *(_DWORD *)(v8 + 256) = *(_DWORD *)(v8 + 24);
      }
      if ( !*(_BYTE *)(v8 + 216) )
        goto LABEL_7;
      if ( v10 )
        break;
      v11 = *(_DWORD *)(v8 + 256);
      v12 = *(_DWORD *)(v8 + 252);
      v10 = 1;
LABEL_7:
      v8 = *(_QWORD *)(v8 + 8);
      if ( v9 == (_BYTE *)v8 )
        goto LABEL_16;
    }
    if ( *(_DWORD *)(v8 + 252) == v12 && *(_DWORD *)(v8 + 256) == v11
      || (_DWORD)qword_501EB30 == v12 && v11 == HIDWORD(qword_501EB30) )
    {
      goto LABEL_7;
    }
    v8 = *(_QWORD *)(v8 + 8);
    v11 = HIDWORD(qword_501EB30);
    v12 = qword_501EB30;
  }
  while ( v9 != (_BYTE *)v8 );
LABEL_16:
  v9 = *(_BYTE **)(a2 + 328);
  if ( v10 && v12 == (_DWORD)qword_501EB30 && HIDWORD(qword_501EB30) == v11 && (_BYTE *)v8 != v9 )
  {
    do
    {
      if ( v9[216] )
      {
        *((_DWORD *)v9 + 63) = v12;
        *((_DWORD *)v9 + 64) = v11;
      }
      v9 = (_BYTE *)*((_QWORD *)v9 + 1);
    }
    while ( (_BYTE *)v8 != v9 );
    v9 = *(_BYTE **)(a2 + 328);
  }
LABEL_18:
  v13 = *(_QWORD *)(v9 + 252);
  v69 = (const __m128i *)v9;
  v14 = 1;
  v63 = v13;
  v68 = &v63;
  v70 = &v64;
  sub_34BCDA0(
    a2,
    (unsigned __int8 (__fastcall *)(__int64, __int64 *, unsigned __int64 *))sub_34BC290,
    (__int64)&v68,
    v5,
    v6,
    v7);
  sub_34BC660(a2);
LABEL_19:
  sub_C7D6A0(v65, 24LL * v67, 4);
  v3 = *(_QWORD *)(a2 + 8);
LABEL_20:
  if ( (*(_BYTE *)(v3 + 879) & 0x10) != 0 )
  {
    v14 = 1;
    sub_2E7A760(a2, 0);
  }
  v15 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501FE44);
  if ( v15 )
  {
    v16 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v15 + 104LL))(v15, &unk_501FE44);
    if ( v16 )
      sub_30052F0(v16 + 200, (__int64)&unk_501FE44, v17, v18, v19, v20);
  }
  v21 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_50209DC);
  if ( v21 )
  {
    v22 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v21 + 104LL))(v21, &unk_50209DC);
    if ( v22 )
      sub_34BD420(v22 + 200, (__int64)&unk_50209DC, v23, v24, v25, v26);
  }
  return v14;
}
