// Function: sub_2EC5A10
// Address: 0x2ec5a10
//
void (*__fastcall sub_2EC5A10(__int64 a1, _BYTE *a2, char a3))()
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 (*v7)(void); // rax
  __int64 v8; // r14
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  __int64 i; // rbx
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r13
  __int64 j; // r13
  __int64 v16; // r8
  __int64 v17; // r9
  __int16 v18; // ax
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 k; // r12
  __int64 v23; // rdx
  const __m128i *v24; // rbx
  unsigned __int64 v25; // rax
  __m128i *v26; // rax
  __int64 v27; // rdx
  const __m128i *v28; // rbx
  const __m128i *v29; // r12
  _QWORD *v30; // r14
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 m; // rax
  _QWORD *v34; // r14
  unsigned __int8 *v35; // rax
  size_t v36; // rdx
  void *v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // r14
  _BYTE *v41; // rax
  const char *v42; // rax
  size_t v43; // rdx
  _WORD *v44; // rdi
  unsigned __int8 *v45; // rsi
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // rdx
  __int64 *v49; // rdi
  void (*result)(); // rax
  unsigned __int64 v51; // rax
  __m128i v52; // xmm2
  __m128i v53; // xmm0
  __int64 v54; // rdx
  __int32 v55; // ecx
  __int64 v56; // rax
  __int64 v57; // rax
  __int8 *v58; // rbx
  __int64 v60; // [rsp+8h] [rbp-228h]
  char v62; // [rsp+17h] [rbp-219h]
  _QWORD *v63; // [rsp+20h] [rbp-210h]
  __int64 v64; // [rsp+28h] [rbp-208h]
  _BYTE *v65; // [rsp+30h] [rbp-200h]
  _QWORD *v66; // [rsp+30h] [rbp-200h]
  size_t v67; // [rsp+30h] [rbp-200h]
  size_t v68; // [rsp+30h] [rbp-200h]
  int v69; // [rsp+48h] [rbp-1E8h]
  char v70; // [rsp+48h] [rbp-1E8h]
  __m128i v71; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v72; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v73; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v74; // [rsp+78h] [rbp-1B8h]
  _BYTE v75[432]; // [rsp+80h] [rbp-1B0h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(v4 + 328);
  v60 = v4 + 320;
  if ( v5 != v4 + 320 )
  {
    while ( 1 )
    {
      v6 = 0;
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a2 + 80LL))(a2, v5);
      v73 = (unsigned __int64)v75;
      v74 = 0x1000000000LL;
      v62 = (*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)a2 + 72LL))(a2);
      v64 = *(_QWORD *)(v5 + 32);
      v7 = *(__int64 (**)(void))(**(_QWORD **)(v64 + 16) + 128LL);
      if ( v7 != sub_2DAC790 )
        v6 = v7();
      v63 = (_QWORD *)(v5 + 48);
      if ( *(_QWORD *)(v5 + 56) != v5 + 48 )
      {
        v65 = a2;
        v8 = v6;
        v9 = v5 + 48;
        do
        {
          if ( v63 != (_QWORD *)v9 )
            goto LABEL_7;
          v47 = *v63 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v47 )
LABEL_96:
            BUG();
          v48 = *(_QWORD *)v47;
          v49 = (__int64 *)(*v63 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_QWORD *)v47 & 4) == 0 && (*(_BYTE *)(v47 + 44) & 4) != 0 )
          {
            while ( 1 )
            {
              v49 = (__int64 *)(v48 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (*(_BYTE *)((v48 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
                break;
              v48 = *v49;
            }
          }
          if ( sub_2EC21B0((__int64)v49, v5, v64, v8) )
          {
LABEL_7:
            v10 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v10 )
              goto LABEL_96;
            v11 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 44) & 4) != 0 )
            {
              for ( i = *(_QWORD *)v10; ; i = *(_QWORD *)v11 )
              {
                v11 = i & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
                  break;
              }
            }
          }
          else
          {
            v11 = v9;
          }
          if ( v11 == *(_QWORD *)(v5 + 56) )
            break;
          v69 = 0;
          v9 = v11;
          while ( 1 )
          {
            v13 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v13 )
              goto LABEL_96;
            v14 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 44) & 4) != 0 )
            {
              for ( j = *(_QWORD *)v13; ; j = *(_QWORD *)v14 )
              {
                v14 = j & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v14 + 44) & 4) == 0 )
                  break;
              }
            }
            if ( sub_2EC21B0(v14, v5, v64, v8) )
              break;
            v18 = *(_WORD *)(v14 + 68);
            if ( (unsigned __int16)(v18 - 14) > 4u )
              v69 += v18 != 24;
            v19 = (_QWORD *)(*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL);
            v20 = v19;
            if ( !v19 )
              goto LABEL_96;
            v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
            v21 = *v19;
            if ( (v21 & 4) == 0 && (*((_BYTE *)v20 + 44) & 4) != 0 )
            {
              for ( k = v21; ; k = *(_QWORD *)v9 )
              {
                v9 = k & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v9 + 44) & 4) == 0 )
                  break;
              }
            }
            if ( *(_QWORD *)(v5 + 56) == v9 )
            {
              if ( v69 )
                goto LABEL_32;
              goto LABEL_35;
            }
          }
          if ( v69 )
          {
LABEL_32:
            v23 = (unsigned int)v74;
            v71.m128i_i64[1] = v11;
            v71.m128i_i64[0] = v9;
            LODWORD(v72) = v69;
            v24 = &v71;
            v25 = v73;
            if ( (unsigned __int64)(unsigned int)v74 + 1 > HIDWORD(v74) )
            {
              if ( v73 > (unsigned __int64)&v71 || (unsigned __int64)&v71 >= v73 + 24LL * (unsigned int)v74 )
              {
                sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 0x18u, v16, v17);
                v25 = v73;
                v23 = (unsigned int)v74;
                v24 = &v71;
              }
              else
              {
                v58 = &v71.m128i_i8[-v73];
                sub_C8D5F0((__int64)&v73, v75, (unsigned int)v74 + 1LL, 0x18u, v16, v17);
                v25 = v73;
                v23 = (unsigned int)v74;
                v24 = (const __m128i *)&v58[v73];
              }
            }
            v26 = (__m128i *)(v25 + 24 * v23);
            *v26 = _mm_loadu_si128(v24);
            v27 = v24[1].m128i_i64[0];
            LODWORD(v74) = v74 + 1;
            v26[1].m128i_i64[0] = v27;
          }
        }
        while ( *(_QWORD *)(v5 + 56) != v9 );
LABEL_35:
        a2 = v65;
      }
      if ( !v62 )
        goto LABEL_37;
      v28 = (const __m128i *)v73;
      v29 = (const __m128i *)(v73 + 24LL * (unsigned int)v74);
      if ( (const __m128i *)v73 != v29 )
        break;
LABEL_74:
      (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)a2 + 88LL))(a2);
      if ( a3 )
        sub_2F92740(a2, v5);
      if ( (_BYTE *)v73 != v75 )
        _libc_free(v73);
      v5 = *(_QWORD *)(v5 + 8);
      if ( v60 == v5 )
        goto LABEL_79;
    }
    v51 = (unsigned __int64)&v29[-2].m128i_u64[1];
    if ( (unsigned __int64)&v29[-2].m128i_u64[1] > v73 )
    {
      do
      {
        v52 = _mm_loadu_si128((const __m128i *)v51);
        v53 = _mm_loadu_si128(v28);
        v51 -= 24LL;
        v28 = (const __m128i *)((char *)v28 + 24);
        v54 = v28[-1].m128i_i64[1];
        *(__m128i *)((char *)v28 - 24) = v52;
        v55 = *(_DWORD *)(v51 + 40);
        v72 = v54;
        v28[-1].m128i_i32[2] = v55;
        v71 = v53;
        *(__m128i *)(v51 + 24) = v53;
        *(_DWORD *)(v51 + 40) = v54;
      }
      while ( (unsigned __int64)v28 < v51 );
LABEL_37:
      v28 = (const __m128i *)v73;
      v70 = a2[897];
      v29 = (const __m128i *)(v73 + 24LL * (unsigned int)v74);
      if ( (const __m128i *)v73 != v29 )
        goto LABEL_42;
      goto LABEL_74;
    }
    v70 = a2[897];
    while ( 1 )
    {
LABEL_42:
      v30 = (_QWORD *)v28->m128i_i64[0];
      v66 = (_QWORD *)v28->m128i_i64[1];
      (*(void (__fastcall **)(_BYTE *, __int64, __int64, _QWORD *, _QWORD))(*(_QWORD *)a2 + 96LL))(
        a2,
        v5,
        v28->m128i_i64[0],
        v66,
        v28[1].m128i_u32[0]);
      if ( v30 == v66 )
        goto LABEL_41;
      if ( v70 )
        break;
      v31 = *v66 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v31 )
        goto LABEL_96;
      v32 = *v66 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 44) & 4) != 0 )
      {
        for ( m = *(_QWORD *)v31; ; m = *(_QWORD *)v32 )
        {
          v32 = m & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v32 + 44) & 4) == 0 )
            break;
        }
      }
      if ( v30 != (_QWORD *)v32 )
      {
        if ( (_BYTE)qword_5021808 )
        {
LABEL_52:
          v34 = sub_CB72A0();
          v35 = (unsigned __int8 *)sub_2E791E0(*(__int64 **)(a1 + 8));
          v37 = (void *)v34[4];
          if ( v34[3] - (_QWORD)v37 < v36 )
          {
            sub_CB6200((__int64)v34, v35, v36);
          }
          else if ( v36 )
          {
            v68 = v36;
            memcpy(v37, v35, v36);
            v34[4] += v68;
          }
          v38 = (__int64)sub_CB72A0();
          v39 = *(_QWORD *)(v38 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(v38 + 24) - v39) <= 5 )
          {
            v38 = sub_CB6200(v38, ":%bb. ", 6u);
          }
          else
          {
            *(_DWORD *)v39 = 1650599226;
            *(_WORD *)(v39 + 4) = 8238;
            *(_QWORD *)(v38 + 32) += 6LL;
          }
          sub_CB59F0(v38, *(int *)(v5 + 24));
          v40 = (__int64)sub_CB72A0();
          v41 = *(_BYTE **)(v40 + 32);
          if ( *(_BYTE **)(v40 + 24) == v41 )
          {
            v40 = sub_CB6200(v40, (unsigned __int8 *)" ", 1u);
          }
          else
          {
            *v41 = 32;
            ++*(_QWORD *)(v40 + 32);
          }
          v42 = sub_2E31BC0(v5);
          v44 = *(_WORD **)(v40 + 32);
          v45 = (unsigned __int8 *)v42;
          v46 = *(_QWORD *)(v40 + 24) - (_QWORD)v44;
          if ( v43 > v46 )
          {
            v57 = sub_CB6200(v40, v45, v43);
            v44 = *(_WORD **)(v57 + 32);
            v40 = v57;
            v46 = *(_QWORD *)(v57 + 24) - (_QWORD)v44;
          }
          else if ( v43 )
          {
            v67 = v43;
            memcpy(v44, v45, v43);
            v56 = *(_QWORD *)(v40 + 24);
            v44 = (_WORD *)(v67 + *(_QWORD *)(v40 + 32));
            *(_QWORD *)(v40 + 32) = v44;
            v46 = v56 - (_QWORD)v44;
          }
          if ( v46 <= 1 )
          {
            sub_CB6200(v40, (unsigned __int8 *)" \n", 2u);
          }
          else
          {
            *v44 = 2592;
            *(_QWORD *)(v40 + 32) += 2LL;
          }
        }
LABEL_40:
        (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)a2 + 112LL))(a2);
      }
LABEL_41:
      v28 = (const __m128i *)((char *)v28 + 24);
      (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)a2 + 104LL))(a2);
      if ( v29 == v28 )
        goto LABEL_74;
    }
    if ( (_BYTE)qword_5021808 )
      goto LABEL_52;
    goto LABEL_40;
  }
LABEL_79:
  result = *(void (**)())(*(_QWORD *)a2 + 120LL);
  if ( result != nullsub_1612 )
    return (void (*)())((__int64 (__fastcall *)(_BYTE *))result)(a2);
  return result;
}
