// Function: sub_980AF0
// Address: 0x980af0
//
__int64 __fastcall sub_980AF0(__int64 a1, _BYTE *a2, size_t a3, _DWORD *a4)
{
  size_t v4; // rdx
  _BYTE *v5; // r15
  __int64 result; // rax
  size_t v7; // rbx
  int v8; // r12d
  __int64 v9; // r13
  int v10; // eax
  int v11; // r8d
  int v12; // ecx
  unsigned int j; // r14d
  __int64 v14; // r12
  const void *v15; // rsi
  bool v16; // al
  unsigned int v17; // r14d
  int v18; // eax
  const __m128i *v19; // r12
  int v20; // r14d
  int v21; // r13d
  int v22; // ebx
  __int64 v23; // r8
  int v24; // r15d
  __int64 v25; // r13
  int v26; // eax
  char *v27; // rdi
  int v28; // r11d
  __int64 v29; // r10
  unsigned int v30; // ecx
  const void *v31; // rsi
  bool v32; // al
  size_t v33; // rdx
  unsigned int v34; // ecx
  int v35; // r13d
  __int64 v36; // r15
  int v37; // eax
  size_t v38; // rdx
  char *v39; // rdi
  int v40; // r11d
  unsigned int i; // r10d
  __int64 v42; // rcx
  bool v43; // al
  const void *v44; // rsi
  unsigned int v45; // r10d
  int v46; // eax
  int *v47; // rcx
  int v48; // eax
  __m128i si128; // xmm0
  int v50; // r15d
  __int64 v51; // r13
  int v52; // eax
  char *v53; // rdi
  int v54; // r11d
  unsigned int v55; // ecx
  bool v56; // al
  const void *v57; // rsi
  size_t v58; // rdx
  unsigned int v59; // ecx
  int v60; // eax
  int v61; // eax
  __int64 v62; // [rsp+0h] [rbp-80h]
  size_t v63; // [rsp+8h] [rbp-78h]
  __int64 v64; // [rsp+8h] [rbp-78h]
  __int64 v65; // [rsp+8h] [rbp-78h]
  __int64 v66; // [rsp+20h] [rbp-60h]
  int v67; // [rsp+20h] [rbp-60h]
  int v68; // [rsp+20h] [rbp-60h]
  int v69; // [rsp+28h] [rbp-58h]
  __int64 v70; // [rsp+28h] [rbp-58h]
  __int64 v71; // [rsp+28h] [rbp-58h]
  unsigned int v72; // [rsp+34h] [rbp-4Ch]
  unsigned int v73; // [rsp+34h] [rbp-4Ch]
  unsigned int v74; // [rsp+34h] [rbp-4Ch]
  int v75; // [rsp+38h] [rbp-48h]
  size_t v76; // [rsp+38h] [rbp-48h]
  int v77; // [rsp+40h] [rbp-40h]
  _BYTE *v78; // [rsp+40h] [rbp-40h]

  v5 = sub_97E150(a2, a3);
  result = 0;
  if ( !v4 )
    return result;
  v7 = v4;
  if ( !byte_4F7FC00 && (unsigned int)sub_2207590(&byte_4F7FC00) )
  {
    qword_4F7FC20 = 1;
    qword_4F7FC28 = 0;
    qword_4F7FC30 = 0;
    dword_4F7FC38 = 0;
    sub_980850((__int64)&qword_4F7FC20, 1024);
    v78 = v5;
    v19 = (const __m128i *)&off_4977320;
    v76 = v7;
    v20 = 0;
    while ( 1 )
    {
      v21 = dword_4F7FC38;
      v22 = v20++;
      if ( !dword_4F7FC38 )
      {
        ++qword_4F7FC20;
        goto LABEL_19;
      }
      v35 = dword_4F7FC38 - 1;
      v36 = qword_4F7FC28;
      v37 = sub_C94890(v19->m128i_i64[0], v19->m128i_i64[1]);
      v38 = v19->m128i_u64[1];
      v39 = (char *)v19->m128i_i64[0];
      v23 = 0;
      v40 = 1;
      for ( i = v35 & v37; ; i = v35 & v45 )
      {
        v42 = v36 + 24LL * i;
        v43 = v39 + 1 == 0;
        v44 = *(const void **)v42;
        if ( *(_QWORD *)v42 != -1 )
        {
          v43 = v39 + 2 == 0;
          if ( v44 != (const void *)-2LL )
          {
            if ( v38 != *(_QWORD *)(v42 + 8) )
              goto LABEL_29;
            v66 = v23;
            v69 = v40;
            v72 = i;
            if ( !v38 )
              goto LABEL_35;
            v62 = v36 + 24LL * i;
            v63 = v38;
            v46 = memcmp(v39, v44, v38);
            v38 = v63;
            v42 = v62;
            i = v72;
            v40 = v69;
            v43 = v46 == 0;
            v23 = v66;
          }
        }
        if ( v43 )
        {
LABEL_35:
          v47 = (int *)(v42 + 16);
          goto LABEL_36;
        }
        if ( v44 == (const void *)-1LL )
          break;
LABEL_29:
        if ( v44 != (const void *)-2LL || v23 )
          v42 = v23;
        v45 = v40 + i;
        v23 = v42;
        ++v40;
      }
      v21 = dword_4F7FC38;
      if ( !v23 )
        v23 = v42;
      ++qword_4F7FC20;
      v48 = qword_4F7FC30 + 1;
      if ( 4 * ((int)qword_4F7FC30 + 1) < (unsigned int)(3 * dword_4F7FC38) )
      {
        if ( dword_4F7FC38 - (v48 + HIDWORD(qword_4F7FC30)) > (unsigned int)dword_4F7FC38 >> 3 )
          goto LABEL_41;
        sub_980850((__int64)&qword_4F7FC20, dword_4F7FC38);
        v23 = 0;
        if ( !dword_4F7FC38 )
          goto LABEL_40;
        v50 = dword_4F7FC38 - 1;
        v51 = qword_4F7FC28;
        v52 = sub_C94890(v19->m128i_i64[0], v19->m128i_i64[1]);
        v53 = (char *)v19->m128i_i64[0];
        v54 = 1;
        v29 = 0;
        v55 = v50 & v52;
        while ( 2 )
        {
          v23 = v51 + 24LL * v55;
          v56 = v53 + 1 == 0;
          v57 = *(const void **)v23;
          if ( *(_QWORD *)v23 != -1 )
          {
            v56 = v53 + 2 == 0;
            if ( v57 != (const void *)-2LL )
            {
              v58 = v19->m128i_u64[1];
              if ( v58 != *(_QWORD *)(v23 + 8) )
              {
LABEL_54:
                if ( v29 || v57 != (const void *)-2LL )
                  v23 = v29;
                v59 = v54 + v55;
                v29 = v23;
                ++v54;
                v55 = v50 & v59;
                continue;
              }
              v67 = v54;
              v70 = v29;
              v73 = v55;
              if ( !v58 )
                goto LABEL_40;
              v64 = v51 + 24LL * v55;
              v60 = memcmp(v53, v57, v58);
              v23 = v64;
              v55 = v73;
              v29 = v70;
              v54 = v67;
              v56 = v60 == 0;
            }
          }
          break;
        }
        if ( v56 )
          goto LABEL_40;
        if ( v57 == (const void *)-1LL )
          goto LABEL_61;
        goto LABEL_54;
      }
LABEL_19:
      sub_980850((__int64)&qword_4F7FC20, 2 * v21);
      v23 = 0;
      if ( !dword_4F7FC38 )
        goto LABEL_40;
      v24 = dword_4F7FC38 - 1;
      v25 = qword_4F7FC28;
      v26 = sub_C94890(v19->m128i_i64[0], v19->m128i_i64[1]);
      v27 = (char *)v19->m128i_i64[0];
      v28 = 1;
      v29 = 0;
      v30 = v24 & v26;
      while ( 2 )
      {
        v23 = v25 + 24LL * v30;
        v31 = *(const void **)v23;
        if ( *(_QWORD *)v23 != -1 )
        {
          v32 = v27 + 2 == 0;
          if ( v31 != (const void *)-2LL )
          {
            v33 = v19->m128i_u64[1];
            if ( *(_QWORD *)(v23 + 8) != v33 )
            {
LABEL_24:
              v34 = v28 + v30;
              ++v28;
              v30 = v24 & v34;
              continue;
            }
            v68 = v28;
            v71 = v29;
            v74 = v30;
            if ( !v33 )
              goto LABEL_40;
            v65 = v25 + 24LL * v30;
            v61 = memcmp(v27, v31, v33);
            v23 = v65;
            v30 = v74;
            v29 = v71;
            v28 = v68;
            v32 = v61 == 0;
          }
          if ( v32 )
            goto LABEL_40;
          if ( !v29 && v31 == (const void *)-2LL )
            v29 = v23;
          goto LABEL_24;
        }
        break;
      }
      if ( v27 == (char *)-1LL )
        goto LABEL_40;
LABEL_61:
      if ( v29 )
        v23 = v29;
LABEL_40:
      v48 = qword_4F7FC30 + 1;
LABEL_41:
      LODWORD(qword_4F7FC30) = v48;
      if ( *(_QWORD *)v23 != -1 )
        --HIDWORD(qword_4F7FC30);
      si128 = _mm_load_si128(v19);
      v47 = (int *)(v23 + 16);
      *(_DWORD *)(v23 + 16) = 0;
      *(__m128i *)v23 = si128;
LABEL_36:
      *v47 = v22;
      ++v19;
      if ( v20 == 523 )
      {
        v5 = v78;
        v7 = v76;
        __cxa_atexit((void (*)(void *))sub_97DF80, &qword_4F7FC20, &qword_4A427C0);
        sub_2207640(&byte_4F7FC00);
        break;
      }
    }
  }
  v8 = dword_4F7FC38;
  if ( !dword_4F7FC38 )
    return 0;
  v9 = qword_4F7FC28;
  v10 = sub_C94890(v5, v7);
  v11 = v8 - 1;
  v12 = 1;
  for ( j = (v8 - 1) & v10; ; j = v11 & v17 )
  {
    v14 = v9 + 24LL * j;
    v15 = *(const void **)v14;
    if ( *(_QWORD *)v14 == -1 )
      break;
    v16 = v5 + 2 == 0;
    if ( v15 != (const void *)-2LL )
    {
      if ( v7 != *(_QWORD *)(v14 + 8) )
        goto LABEL_8;
      v75 = v12;
      v77 = v11;
      v18 = memcmp(v5, v15, v7);
      v11 = v77;
      v12 = v75;
      v16 = v18 == 0;
    }
    if ( v16 )
      goto LABEL_11;
LABEL_8:
    v17 = v12 + j;
    ++v12;
  }
  if ( v5 != (_BYTE *)-1LL )
    return 0;
LABEL_11:
  if ( v14 == qword_4F7FC28 + 24LL * (unsigned int)dword_4F7FC38 )
    return 0;
  *a4 = *(_DWORD *)(v14 + 16);
  return 1;
}
