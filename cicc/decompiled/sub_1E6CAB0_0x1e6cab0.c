// Function: sub_1E6CAB0
// Address: 0x1e6cab0
//
void (*__fastcall sub_1E6CAB0(__int64 a1, __int64 a2, char a3))()
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  int v7; // r8d
  __int64 (*v8)(void); // rax
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r9
  __int64 i; // r9
  __int64 v15; // r12
  __int64 v16; // r15
  int v17; // r13d
  unsigned __int64 v18; // rbx
  __int16 v19; // ax
  __int64 j; // rbx
  __int64 v21; // rax
  int v22; // eax
  int v23; // r12d
  __int64 v24; // rax
  __m128i *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  const __m128i *v28; // rbx
  _QWORD *v29; // r13
  _QWORD *v30; // r14
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rax
  __int64 m; // rax
  void (*result)(); // rax
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  __int64 v37; // rax
  __int64 k; // r14
  unsigned __int64 v39; // rbx
  __int64 v40; // rdx
  __int16 v41; // ax
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  char v45; // al
  _QWORD *v46; // r13
  char *v47; // rax
  size_t v48; // rdx
  void *v49; // rdi
  size_t v50; // r14
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // r13
  _BYTE *v54; // rax
  const char *v55; // rax
  size_t v56; // rdx
  _WORD *v57; // rdi
  char *v58; // rsi
  size_t v59; // r14
  unsigned __int64 v60; // rax
  const __m128i *v61; // rax
  unsigned __int64 v62; // rax
  __m128i v63; // xmm2
  __m128i v64; // xmm0
  __int64 v65; // rdx
  __int32 v66; // ecx
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // [rsp+8h] [rbp-228h]
  __int64 v71; // [rsp+20h] [rbp-210h]
  char v73; // [rsp+2Fh] [rbp-201h]
  _QWORD *v74; // [rsp+30h] [rbp-200h]
  __int64 v75; // [rsp+40h] [rbp-1F0h]
  unsigned __int64 v76; // [rsp+48h] [rbp-1E8h]
  __m128i v77; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v78; // [rsp+60h] [rbp-1D0h]
  const __m128i *v79; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v80; // [rsp+78h] [rbp-1B8h]
  _BYTE v81[432]; // [rsp+80h] [rbp-1B0h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(v4 + 328);
  v71 = v4 + 320;
  if ( v5 == v4 + 320 )
    goto LABEL_48;
  do
  {
    v6 = 0;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 72LL))(a2, v5);
    v79 = (const __m128i *)v81;
    v80 = 0x1000000000LL;
    v73 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 64LL))(a2);
    v75 = *(_QWORD *)(v5 + 56);
    v8 = *(__int64 (**)(void))(**(_QWORD **)(v75 + 16) + 40LL);
    if ( v8 != sub_1D00B00 )
      v6 = v8();
    v9 = v5 + 24;
    v74 = (_QWORD *)(v5 + 24);
    if ( v5 + 24 != *(_QWORD *)(v5 + 32) )
    {
      v69 = a2;
      v10 = v6;
      v11 = v5;
      while ( 1 )
      {
        if ( v74 != (_QWORD *)v9 )
        {
LABEL_7:
          v12 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v12 )
            goto LABEL_97;
          v13 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)v12 & 4) == 0 && (*(_BYTE *)(v12 + 46) & 4) != 0 )
          {
            for ( i = *(_QWORD *)v12; ; i = *(_QWORD *)v13 )
            {
              v13 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v13 + 46) & 4) == 0 )
                break;
            }
          }
          v9 = *(_QWORD *)(v11 + 32);
          if ( v13 != v9 )
          {
LABEL_14:
            v76 = v13;
            v9 = v13;
            v15 = v10;
            v16 = v11;
            v17 = 0;
            while ( 1 )
            {
              v18 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v18 )
                goto LABEL_97;
              v19 = *(_WORD *)(v18 + 46);
              if ( (*(_QWORD *)v18 & 4) != 0 )
              {
                if ( (v19 & 4) != 0 )
                {
LABEL_59:
                  v21 = (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 8LL) >> 4) & 1LL;
                  goto LABEL_23;
                }
              }
              else if ( (v19 & 4) != 0 )
              {
                for ( j = *(_QWORD *)v18; ; j = *(_QWORD *)v18 )
                {
                  v18 = j & 0xFFFFFFFFFFFFFFF8LL;
                  v19 = *(_WORD *)(v18 + 46);
                  if ( (v19 & 4) == 0 )
                    break;
                }
              }
              if ( (v19 & 8) == 0 )
                goto LABEL_59;
              LOBYTE(v21) = sub_1E15D00(v18, 0x10u, 1);
LABEL_23:
              if ( !(_BYTE)v21
                && !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64, __int64, __int64))(*(_QWORD *)v15 + 736LL))(
                      v15,
                      v18,
                      v16,
                      v75) )
              {
                v17 -= ((unsigned __int16)(**(_WORD **)(v18 + 16) - 12) < 2u) - 1;
                v35 = (_QWORD *)(*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL);
                v36 = v35;
                if ( !v35 )
                  goto LABEL_97;
                v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
                v37 = *v35;
                if ( (v37 & 4) == 0 && (*((_BYTE *)v36 + 46) & 4) != 0 )
                {
                  for ( k = v37; ; k = *(_QWORD *)v9 )
                  {
                    v9 = k & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (*(_BYTE *)(v9 + 46) & 4) == 0 )
                      break;
                  }
                }
                if ( *(_QWORD *)(v16 + 32) != v9 )
                  continue;
              }
              v22 = v17;
              v13 = v76;
              v11 = v16;
              v10 = v15;
              v23 = v22;
              goto LABEL_26;
            }
          }
          goto LABEL_71;
        }
        v39 = *v74 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v39 )
LABEL_97:
          BUG();
        v40 = *(_QWORD *)v39;
        v41 = *(_WORD *)(v39 + 46);
        v42 = *v74 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v39 & 4) != 0 )
        {
          if ( (v41 & 4) != 0 )
          {
LABEL_84:
            v44 = (*(_QWORD *)(*(_QWORD *)(v42 + 16) + 8LL) >> 4) & 1LL;
            goto LABEL_68;
          }
        }
        else if ( (v41 & 4) != 0 )
        {
          while ( 1 )
          {
            v43 = v40 & 0xFFFFFFFFFFFFFFF8LL;
            v41 = *(_WORD *)(v43 + 46);
            v42 = v43;
            if ( (v41 & 4) == 0 )
              break;
            v40 = *(_QWORD *)v43;
          }
        }
        if ( (v41 & 8) == 0 )
          goto LABEL_84;
        LOBYTE(v44) = sub_1E15D00(v42, 0x10u, 1);
LABEL_68:
        if ( (_BYTE)v44 )
          goto LABEL_7;
        v45 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64))(*(_QWORD *)v10 + 736LL))(
                v10,
                v42,
                v11,
                v75);
        v13 = v9;
        if ( v45 )
          goto LABEL_7;
        v9 = *(_QWORD *)(v11 + 32);
        if ( v13 != v9 )
          goto LABEL_14;
LABEL_71:
        v23 = 0;
LABEL_26:
        v77.m128i_i64[0] = v9;
        v24 = (unsigned int)v80;
        v77.m128i_i64[1] = v13;
        LODWORD(v78) = v23;
        if ( (unsigned int)v80 >= HIDWORD(v80) )
        {
          sub_16CD150((__int64)&v79, v81, 0, 24, v7, v13);
          v24 = (unsigned int)v80;
        }
        v25 = (__m128i *)((char *)v79 + 24 * v24);
        v26 = v78;
        *v25 = _mm_loadu_si128(&v77);
        v25[1].m128i_i64[0] = v26;
        v27 = (unsigned int)(v80 + 1);
        LODWORD(v80) = v80 + 1;
        if ( *(_QWORD *)(v11 + 32) == v9 )
        {
          a2 = v69;
          v5 = v11;
          goto LABEL_30;
        }
      }
    }
    v27 = (unsigned int)v80;
LABEL_30:
    if ( !v73 )
      goto LABEL_31;
    v28 = v79;
    v61 = (const __m128i *)((char *)v79 + 24 * v27);
    if ( v79 == v61 )
      goto LABEL_43;
    v62 = (unsigned __int64)&v61[-2].m128i_u64[1];
    if ( (unsigned __int64)v79 >= v62 )
    {
      while ( 1 )
      {
LABEL_32:
        v29 = (_QWORD *)v28->m128i_i64[0];
        v30 = (_QWORD *)v28->m128i_i64[1];
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD *, _QWORD))(*(_QWORD *)a2 + 88LL))(
          a2,
          v5,
          v28->m128i_i64[0],
          v30,
          v28[1].m128i_u32[0]);
        if ( v29 == v30 )
          goto LABEL_42;
        v31 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v31 )
          goto LABEL_97;
        v32 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 46) & 4) != 0 )
        {
          for ( m = *(_QWORD *)v31; ; m = *(_QWORD *)v32 )
          {
            v32 = m & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v32 + 46) & 4) == 0 )
              break;
          }
        }
        if ( v29 == (_QWORD *)v32 )
          goto LABEL_42;
        if ( LOBYTE(qword_4FC7CE0[20]) )
          break;
LABEL_41:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a2 + 104LL))(a2);
LABEL_42:
        v28 = (const __m128i *)((char *)v28 + 24);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a2 + 96LL))(a2);
        if ( v28 == (const __m128i *)((char *)v79 + 24 * (unsigned int)v80) )
          goto LABEL_43;
      }
      v46 = sub_16E8CB0();
      v47 = (char *)sub_1E0A440(*(__int64 **)(a1 + 8));
      v49 = (void *)v46[3];
      v50 = v48;
      if ( v48 > v46[2] - (_QWORD)v49 )
      {
        sub_16E7EE0((__int64)v46, v47, v48);
      }
      else if ( v48 )
      {
        memcpy(v49, v47, v48);
        v46[3] += v50;
      }
      v51 = (__int64)sub_16E8CB0();
      v52 = *(_QWORD *)(v51 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v51 + 16) - v52) <= 5 )
      {
        v51 = sub_16E7EE0(v51, ":%bb. ", 6u);
      }
      else
      {
        *(_DWORD *)v52 = 1650599226;
        *(_WORD *)(v52 + 4) = 8238;
        *(_QWORD *)(v51 + 24) += 6LL;
      }
      sub_16E7AB0(v51, *(int *)(v5 + 48));
      v53 = (__int64)sub_16E8CB0();
      v54 = *(_BYTE **)(v53 + 24);
      if ( *(_BYTE **)(v53 + 16) == v54 )
      {
        v53 = sub_16E7EE0(v53, " ", 1u);
      }
      else
      {
        *v54 = 32;
        ++*(_QWORD *)(v53 + 24);
      }
      v55 = sub_1DD6290(v5);
      v57 = *(_WORD **)(v53 + 24);
      v58 = (char *)v55;
      v59 = v56;
      v60 = *(_QWORD *)(v53 + 16) - (_QWORD)v57;
      if ( v60 < v56 )
      {
        v68 = sub_16E7EE0(v53, v58, v56);
        v57 = *(_WORD **)(v68 + 24);
        v53 = v68;
        v60 = *(_QWORD *)(v68 + 16) - (_QWORD)v57;
      }
      else if ( v56 )
      {
        memcpy(v57, v58, v56);
        v57 = (_WORD *)(v59 + *(_QWORD *)(v53 + 24));
        v67 = *(_QWORD *)(v53 + 16) - (_QWORD)v57;
        *(_QWORD *)(v53 + 24) = v57;
        if ( v67 <= 1 )
        {
LABEL_93:
          sub_16E7EE0(v53, " \n", 2u);
          goto LABEL_41;
        }
        goto LABEL_82;
      }
      if ( v60 <= 1 )
        goto LABEL_93;
LABEL_82:
      *v57 = 2592;
      *(_QWORD *)(v53 + 24) += 2LL;
      goto LABEL_41;
    }
    do
    {
      v63 = _mm_loadu_si128((const __m128i *)v62);
      v64 = _mm_loadu_si128(v28);
      v62 -= 24LL;
      v28 = (const __m128i *)((char *)v28 + 24);
      v65 = v28[-1].m128i_i64[1];
      *(__m128i *)((char *)v28 - 24) = v63;
      v66 = *(_DWORD *)(v62 + 40);
      v78 = v65;
      v28[-1].m128i_i32[2] = v66;
      v77 = v64;
      *(__m128i *)(v62 + 24) = v64;
      *(_DWORD *)(v62 + 40) = v65;
    }
    while ( (unsigned __int64)v28 < v62 );
    v27 = (unsigned int)v80;
LABEL_31:
    v28 = v79;
    if ( 3 * v27 )
      goto LABEL_32;
LABEL_43:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a2 + 80LL))(a2);
    if ( a3 )
      sub_1F04A30(a2, v5);
    if ( v79 != (const __m128i *)v81 )
      _libc_free((unsigned __int64)v79);
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v71 != v5 );
LABEL_48:
  result = *(void (**)())(*(_QWORD *)a2 + 112LL);
  if ( result != nullsub_706 )
    return (void (*)())((__int64 (__fastcall *)(__int64))result)(a2);
  return result;
}
