// Function: sub_3542B20
// Address: 0x3542b20
//
void __fastcall sub_3542B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rcx
  int v16; // edi
  __int64 v17; // r10
  int v18; // edi
  unsigned int v19; // esi
  unsigned __int64 *v20; // rdx
  unsigned __int64 v21; // rsi
  int v22; // eax
  void (*v23)(); // rbx
  __int64 v24; // rcx
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // r9
  unsigned int v27; // r12d
  __int64 v28; // rdi
  __int64 v29; // rbx
  __int64 v30; // rax
  int v31; // esi
  __int64 v32; // rdi
  int v33; // esi
  unsigned int v34; // ecx
  unsigned __int64 *v35; // rdx
  unsigned __int64 v36; // rdx
  __int64 v37; // rcx
  _QWORD *v38; // rcx
  _QWORD *j; // rdi
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  int i; // edx
  const __m128i *v43; // rbx
  __int64 v44; // rax
  const __m128i *k; // r12
  __int64 v46; // rcx
  __int64 v47; // rdi
  int v48; // esi
  __int64 v49; // rdx
  int v50; // edx
  __m128i v51; // xmm0
  unsigned __int64 v52; // r12
  _BYTE *v53; // rbx
  __m128i *v54; // rsi
  int v55; // edx
  __int64 v56; // [rsp+10h] [rbp-D0h]
  __int64 v57; // [rsp+20h] [rbp-C0h]
  __m128i v58; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+40h] [rbp-A0h]
  unsigned int v60; // [rsp+48h] [rbp-98h]
  unsigned int v61; // [rsp+4Ch] [rbp-94h]
  unsigned __int64 v62; // [rsp+50h] [rbp-90h] BYREF
  __int64 v63; // [rsp+58h] [rbp-88h]
  _BYTE *v64; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v65; // [rsp+68h] [rbp-78h]
  unsigned int v66; // [rsp+6Ch] [rbp-74h]
  _BYTE v67[112]; // [rsp+70h] [rbp-70h] BYREF

  v64 = v67;
  v6 = *(_QWORD *)(a1 + 48);
  v7 = *(_QWORD *)(a1 + 32);
  v66 = 4;
  v57 = *(_QWORD *)(v7 + 16);
  v56 = *(_QWORD *)(a1 + 56);
  if ( v6 == v56 )
    return;
  v8 = v6;
  do
  {
    v65 = 0;
    v10 = *(_QWORD *)v8;
    v61 = 0;
    v11 = *(_QWORD *)(v10 + 32);
    v59 = v10;
    v60 = 0;
    v58.m128i_i64[0] = v11 + 40LL * (*(_DWORD *)(v10 + 40) & 0xFFFFFF);
    if ( v58.m128i_i64[0] == v11 )
      goto LABEL_40;
    v12 = v8;
    v13 = v11;
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v13 )
        {
          v27 = *(_DWORD *)(v13 + 8);
          v28 = *(_QWORD *)(a1 + 40);
          if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
          {
            v14 = sub_2EBEE90(v28, v27);
            if ( v14 )
            {
              v16 = *(_DWORD *)(a1 + 960);
              v17 = *(_QWORD *)(a1 + 944);
              if ( v16 )
              {
                v18 = v16 - 1;
                v19 = v18 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
                v20 = (unsigned __int64 *)(v17 + 16LL * v19);
                a6 = *v20;
                if ( v14 == *v20 )
                {
LABEL_8:
                  v21 = v20[1];
                  if ( v21 )
                  {
                    v22 = *(unsigned __int16 *)(v14 + 68);
                    if ( !v22 || v22 == 68 )
                    {
                      if ( !*(_WORD *)(v59 + 68) || *(_WORD *)(v59 + 68) == 68 )
                      {
                        v60 = v27;
                        if ( *(_DWORD *)(v21 + 200) < *(_DWORD *)(v12 + 200) )
                        {
                          v40 = *(_QWORD **)(v12 + 40);
                          v41 = &v40[2 * *(unsigned int *)(v12 + 48)];
                          if ( v40 == v41 )
                          {
LABEL_89:
                            v63 = 0;
                            v62 = v21 | 6;
                            sub_2F8F1B0(v12, (__int64)&v62, 1u, v15, a5, a6);
                            v60 = v27;
                          }
                          else
                          {
                            while ( 1 )
                            {
                              v15 = *v40 & 0xFFFFFFFFFFFFFFF8LL;
                              if ( v21 == v15 )
                                break;
                              v40 += 2;
                              if ( v41 == v40 )
                                goto LABEL_89;
                            }
                            v60 = v27;
                          }
                        }
                      }
                      else
                      {
                        v63 = v27;
                        v62 = v21 & 0xFFFFFFFFFFFFFFF9LL;
                        v23 = *(void (**)())(*(_QWORD *)v57 + 344LL);
                        v25 = (unsigned int)sub_2EAB0A0(v13);
                        if ( v23 != nullsub_1667 )
                          ((void (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, unsigned __int64, unsigned __int64 *, __int64))v23)(
                            v57,
                            v21,
                            0,
                            v12,
                            v25,
                            &v62,
                            a1 + 600);
                        sub_2F8F1B0(v12, (__int64)&v62, 1u, v24, v25, v26);
                      }
                    }
                  }
                }
                else
                {
                  v55 = 1;
                  while ( a6 != -4096 )
                  {
                    v15 = (unsigned int)(v55 + 1);
                    v19 = v18 & (v55 + v19);
                    v20 = (unsigned __int64 *)(v17 + 16LL * v19);
                    a6 = *v20;
                    if ( v14 == *v20 )
                      goto LABEL_8;
                    v55 = v15;
                  }
                }
              }
            }
            goto LABEL_16;
          }
          v29 = (v27 & 0x80000000) != 0
              ? *(_QWORD *)(*(_QWORD *)(v28 + 56) + 16LL * (v27 & 0x7FFFFFFF) + 8)
              : *(_QWORD *)(*(_QWORD *)(v28 + 304) + 8LL * v27);
          if ( v29 )
            break;
        }
LABEL_16:
        v13 += 40;
        if ( v58.m128i_i64[0] == v13 )
          goto LABEL_39;
      }
      if ( (*(_BYTE *)(v29 + 3) & 0x10) == 0 )
      {
LABEL_23:
        v30 = *(_QWORD *)(v29 + 16);
LABEL_24:
        v31 = *(_DWORD *)(a1 + 960);
        v32 = *(_QWORD *)(a1 + 944);
        if ( !v31 )
          goto LABEL_32;
        v33 = v31 - 1;
        v34 = v33 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v35 = (unsigned __int64 *)(v32 + 16LL * v34);
        a5 = *v35;
        if ( *v35 != v30 )
        {
          for ( i = 1; ; i = a6 )
          {
            if ( a5 == -4096 )
              goto LABEL_32;
            a6 = (unsigned int)(i + 1);
            v34 = v33 & (v34 + i);
            v35 = (unsigned __int64 *)(v32 + 16LL * v34);
            a5 = *v35;
            if ( *v35 == v30 )
              break;
          }
        }
        v36 = v35[1];
        if ( v36 && (!*(_WORD *)(v30 + 68) || *(_WORD *)(v30 + 68) == 68) )
        {
          v37 = *(unsigned __int16 *)(v59 + 68);
          if ( *(_WORD *)(v59 + 68) && (_DWORD)v37 != 68 )
          {
            v63 = v27 | 0x100000000LL;
            v62 = v36 & 0xFFFFFFFFFFFFFFF9LL | 2;
            sub_2F8F1B0(v12, (__int64)&v62, 1u, v37, a5, a6);
            v30 = *(_QWORD *)(v29 + 16);
            goto LABEL_32;
          }
          v61 = v27;
          if ( *(_DWORD *)(v36 + 200) < *(_DWORD *)(v12 + 200) )
          {
            v38 = *(_QWORD **)(v12 + 40);
            for ( j = &v38[2 * *(unsigned int *)(v12 + 48)]; j != v38; v38 += 2 )
            {
              if ( v36 == (*v38 & 0xFFFFFFFFFFFFFFF8LL) )
              {
                v61 = v27;
                goto LABEL_32;
              }
            }
            v63 = 0;
            v62 = v36 | 6;
            sub_2F8F1B0(v12, (__int64)&v62, 1u, (__int64)v38, a5, a6);
            v61 = v27;
            v30 = *(_QWORD *)(v29 + 16);
          }
        }
LABEL_32:
        while ( 1 )
        {
          v29 = *(_QWORD *)(v29 + 32);
          if ( !v29 )
            goto LABEL_16;
          if ( (*(_BYTE *)(v29 + 3) & 0x10) == 0 && *(_QWORD *)(v29 + 16) != v30 )
          {
            v30 = *(_QWORD *)(v29 + 16);
            goto LABEL_24;
          }
        }
      }
      while ( 1 )
      {
        v29 = *(_QWORD *)(v29 + 32);
        if ( !v29 )
          break;
        if ( (*(_BYTE *)(v29 + 3) & 0x10) == 0 )
          goto LABEL_23;
      }
      v13 += 40;
    }
    while ( v58.m128i_i64[0] != v13 );
LABEL_39:
    v8 = v12;
LABEL_40:
    if ( (_BYTE)qword_503EA48 )
    {
      v43 = *(const __m128i **)(v8 + 40);
      v44 = v65;
      for ( k = &v43[*(unsigned int *)(v8 + 48)]; k != v43; ++v65 )
      {
        while ( 1 )
        {
          v46 = *(_QWORD *)(v43->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_WORD *)(v46 + 68) == 68 || !*(_WORD *)(v46 + 68))
            && (((unsigned __int8)v43->m128i_i64[0] ^ 6) & 6) == 0 )
          {
            if ( *(_WORD *)(*(_QWORD *)v8 + 68LL) != 68 && *(_WORD *)(*(_QWORD *)v8 + 68LL) )
              break;
            v47 = *(_QWORD *)(v46 + 32);
            if ( *(_DWORD *)(v47 + 8) != v60 )
            {
              a5 = *(_QWORD *)(v46 + 24);
              v48 = *(_DWORD *)(v46 + 40) & 0xFFFFFF;
              if ( v48 == 1 )
              {
LABEL_88:
                v50 = 0;
              }
              else
              {
                v49 = 1;
                while ( a5 != *(_QWORD *)(v47 + 40LL * (unsigned int)(v49 + 1) + 24) )
                {
                  v49 = (unsigned int)(v49 + 2);
                  if ( v48 == (_DWORD)v49 )
                    goto LABEL_88;
                }
                v50 = *(_DWORD *)(v47 + 40 * v49 + 8);
              }
              if ( v50 != v61 )
                break;
            }
          }
          if ( k == ++v43 )
            goto LABEL_80;
        }
        v51 = _mm_loadu_si128(v43);
        if ( v44 + 1 > (unsigned __int64)v66 )
        {
          v58 = v51;
          sub_C8D5F0((__int64)&v64, v67, v44 + 1, 0x10u, a5, a6);
          v44 = v65;
          v51 = _mm_load_si128(&v58);
        }
        ++v43;
        *(__m128i *)&v64[16 * v44] = v51;
        v44 = v65 + 1;
      }
LABEL_80:
      v52 = (unsigned __int64)v64;
      v53 = &v64[16 * (unsigned int)v44];
      if ( v53 != v64 )
      {
        do
        {
          v54 = (__m128i *)v52;
          v52 += 16LL;
          sub_2F8F420(v8, v54);
        }
        while ( v53 != (_BYTE *)v52 );
      }
    }
    v8 += 256;
  }
  while ( v56 != v8 );
  if ( v64 != v67 )
    _libc_free((unsigned __int64)v64);
}
