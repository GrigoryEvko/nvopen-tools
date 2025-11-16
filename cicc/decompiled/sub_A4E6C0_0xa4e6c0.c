// Function: sub_A4E6C0
// Address: 0xa4e6c0
//
_BYTE *__fastcall sub_A4E6C0(_BYTE *a1, __int64 a2, char a3)
{
  __int64 v3; // r15
  char **v4; // r13
  _BYTE *v5; // r12
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  volatile signed __int32 *v12; // rdi
  signed __int32 v13; // eax
  unsigned __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  char v19; // al
  const __m128i *v20; // rax
  const __m128i *v21; // rax
  const __m128i *v22; // rax
  const __m128i *v23; // r14
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rdi
  const __m128i *v27; // rdi
  __int64 v28; // r12
  __int64 v29; // r13
  volatile signed __int32 *v30; // r15
  signed __int32 v31; // eax
  signed __int32 v32; // eax
  __int64 v33; // rax
  unsigned int v34; // edx
  char *v35; // r8
  char *v36; // rcx
  __int64 v37; // rax
  char *v38; // r12
  char *v39; // rbx
  volatile signed __int32 *v40; // r13
  char v41; // al
  int v42; // edx
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // rbx
  volatile signed __int32 *v47; // r13
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdx
  const __m128i *v51; // rax
  __int64 v52; // rdi
  bool v53; // dl
  __int64 *v54; // rdi
  __int64 v55; // rdx
  signed __int32 v56; // eax
  const __m128i *v57; // rdi
  unsigned __int64 v58; // rax
  __int64 v59; // rdx
  char **v60; // [rsp+8h] [rbp-2E8h]
  char v61; // [rsp+10h] [rbp-2E0h]
  char **v62; // [rsp+10h] [rbp-2E0h]
  _BYTE *v63; // [rsp+18h] [rbp-2D8h]
  char v64; // [rsp+18h] [rbp-2D8h]
  char *src; // [rsp+20h] [rbp-2D0h]
  _BYTE *srca; // [rsp+20h] [rbp-2D0h]
  __int64 v67; // [rsp+28h] [rbp-2C8h]
  _QWORD *v68; // [rsp+28h] [rbp-2C8h]
  _BYTE *v69; // [rsp+30h] [rbp-2C0h]
  __int64 *v70; // [rsp+38h] [rbp-2B8h]
  const __m128i *v71; // [rsp+38h] [rbp-2B8h]
  __int64 v72; // [rsp+38h] [rbp-2B8h]
  unsigned __int64 v73; // [rsp+40h] [rbp-2B0h] BYREF
  char v74; // [rsp+48h] [rbp-2A8h]
  unsigned __int64 v75; // [rsp+50h] [rbp-2A0h] BYREF
  unsigned __int64 v76; // [rsp+60h] [rbp-290h] BYREF
  char v77; // [rsp+68h] [rbp-288h]
  const __m128i *v78; // [rsp+70h] [rbp-280h] BYREF
  const __m128i *v79; // [rsp+78h] [rbp-278h]
  const __m128i *v80; // [rsp+80h] [rbp-270h]
  __m128i v81; // [rsp+90h] [rbp-260h] BYREF
  __m128i v82; // [rsp+A0h] [rbp-250h] BYREF
  __int64 v83; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-238h]
  _BYTE v85[560]; // [rsp+C0h] [rbp-230h] BYREF

  v3 = a2;
  v4 = (char **)&v83;
  v5 = a1;
  sub_A4DCE0(&v83, a2, 0, 0);
  v8 = v83 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v83 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    a1[32] |= 3u;
    *(_QWORD *)a1 = v8;
    return v5;
  }
  v78 = 0;
  v83 = (__int64)v85;
  v79 = 0;
  v80 = 0;
  v84 = 0x4000000000LL;
  v70 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
LABEL_3:
        if ( !*(_DWORD *)(v3 + 32) && *(_QWORD *)(v3 + 8) <= *(_QWORD *)(v3 + 16) )
        {
          v75 = 0;
          v9 = 0;
          break;
        }
        a2 = v3;
        sub_9C66D0((__int64)&v81, v3, *(unsigned int *)(v3 + 36), v7);
        if ( (v81.m128i_i8[8] & 1) != 0 )
        {
          v76 = 0;
          v77 = v81.m128i_i8[8] & 1 | v77 & 0xFC;
          v75 = v81.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
LABEL_21:
          v73 = v75;
LABEL_28:
          v17 = v73;
          v5[32] |= 3u;
          *(_QWORD *)v5 = v17 & 0xFFFFFFFFFFFFFFFELL;
          goto LABEL_32;
        }
        v16 = v81.m128i_i64[0];
        v77 = v81.m128i_i8[8] & 1 | v77 & 0xFC;
        LODWORD(v76) = v81.m128i_i32[0];
        if ( v81.m128i_i32[0] )
        {
          if ( v81.m128i_i32[0] == 1 )
          {
            a2 = v3;
            sub_9CE2D0((__int64)&v81, v3, 8, v15);
            if ( (v81.m128i_i8[8] & 1) != 0 )
            {
              v81.m128i_i8[8] &= ~2u;
              v49 = v81.m128i_i64[0];
              v81.m128i_i64[0] = 0;
              v75 = v49 & 0xFFFFFFFFFFFFFFFELL;
              goto LABEL_21;
            }
            LODWORD(v75) = 2;
            HIDWORD(v75) = v81.m128i_i32[0];
          }
          else
          {
            LODWORD(v75) = 3;
            HIDWORD(v75) = v81.m128i_i32[0];
          }
        }
        else
        {
          v33 = *(unsigned int *)(v3 + 72);
          if ( (_DWORD)v33 )
          {
            v34 = *(_DWORD *)(v3 + 32);
            if ( v34 > 0x1F )
            {
              *(_DWORD *)(v3 + 32) = 32;
              *(_QWORD *)(v3 + 24) >>= (unsigned __int8)v34 - 32;
            }
            else
            {
              *(_DWORD *)(v3 + 32) = 0;
            }
            v35 = *(char **)(v3 + 40);
            a2 = *(_QWORD *)(v3 + 56);
            v36 = *(char **)(v3 + 48);
            v37 = *(_QWORD *)(v3 + 64) + 32 * v33 - 32;
            v67 = a2;
            *(_DWORD *)(v3 + 36) = *(_DWORD *)v37;
            *(_QWORD *)(v3 + 40) = *(_QWORD *)(v37 + 8);
            *(_QWORD *)(v3 + 48) = *(_QWORD *)(v37 + 16);
            v16 = *(_QWORD *)(v37 + 24);
            *(_QWORD *)(v3 + 56) = v16;
            *(_QWORD *)(v37 + 8) = 0;
            *(_QWORD *)(v37 + 16) = 0;
            *(_QWORD *)(v37 + 24) = 0;
            if ( v35 != v36 )
            {
              v63 = v5;
              v38 = v35;
              v61 = a3;
              v39 = v36;
              src = v35;
              v60 = v4;
              do
              {
                v40 = (volatile signed __int32 *)*((_QWORD *)v38 + 1);
                if ( v40 )
                {
                  if ( &_pthread_key_create )
                  {
                    v16 = (unsigned int)_InterlockedExchangeAdd(v40 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v16 = *((unsigned int *)v40 + 2);
                    *((_DWORD *)v40 + 2) = v16 - 1;
                  }
                  if ( (_DWORD)v16 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v40 + 16LL))(v40);
                    if ( &_pthread_key_create )
                    {
                      v16 = (unsigned int)_InterlockedExchangeAdd(v40 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v16 = *((unsigned int *)v40 + 3);
                      a2 = (unsigned int)(v16 - 1);
                      *((_DWORD *)v40 + 3) = a2;
                    }
                    if ( (_DWORD)v16 == 1 )
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v40 + 24LL))(v40);
                  }
                }
                v38 += 16;
              }
              while ( v39 != v38 );
              v35 = src;
              v5 = v63;
              a3 = v61;
              v4 = v60;
            }
            if ( v35 )
            {
              a2 = v67 - (_QWORD)v35;
              j_j___libc_free_0(v35, v67 - (_QWORD)v35);
            }
            v44 = (unsigned int)(*(_DWORD *)(v3 + 72) - 1);
            *(_DWORD *)(v3 + 72) = v44;
            v68 = (_QWORD *)(*(_QWORD *)(v3 + 64) + 32 * v44);
            v15 = v68[2];
            if ( v15 != v68[1] )
            {
              srca = v5;
              v45 = v68[2];
              v64 = a3;
              v46 = v68[1];
              v62 = v4;
              do
              {
                v47 = *(volatile signed __int32 **)(v46 + 8);
                if ( v47 )
                {
                  if ( &_pthread_key_create )
                  {
                    v16 = (unsigned int)_InterlockedExchangeAdd(v47 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v16 = *((unsigned int *)v47 + 2);
                    *((_DWORD *)v47 + 2) = v16 - 1;
                  }
                  if ( (_DWORD)v16 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v47 + 16LL))(v47);
                    if ( &_pthread_key_create )
                    {
                      v16 = (unsigned int)_InterlockedExchangeAdd(v47 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v16 = *((unsigned int *)v47 + 3);
                      a2 = (unsigned int)(v16 - 1);
                      *((_DWORD *)v47 + 3) = a2;
                    }
                    if ( (_DWORD)v16 == 1 )
                      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v47 + 24LL))(v47);
                  }
                }
                v46 += 16;
              }
              while ( v45 != v46 );
              v5 = srca;
              a3 = v64;
              v4 = v62;
            }
            v48 = v68[1];
            if ( v48 )
            {
              a2 = v68[3] - v48;
              j_j___libc_free_0(v48, a2);
            }
            v75 = 1;
            if ( (v77 & 2) != 0 )
              sub_9CE230(&v76);
            if ( (v77 & 1) != 0 && v76 )
              (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v76 + 8LL))(v76);
          }
          else
          {
            v75 = 0;
          }
        }
        v9 = v75;
        if ( (_DWORD)v75 != 2 )
          break;
        a2 = v3;
        sub_9CE5C0(v81.m128i_i64, v3, v16, v15);
        if ( (v81.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v73 = v81.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
          goto LABEL_28;
        }
      }
      v74 &= 0xFCu;
      v73 = __PAIR64__(HIDWORD(v75), v9);
      if ( v9 == 1 )
      {
        v19 = v5[32];
        v5[24] = 1;
        v5[32] = v19 & 0xFC | 2;
        v20 = v78;
        v78 = 0;
        *(_QWORD *)v5 = v20;
        v21 = v79;
        v79 = 0;
        *((_QWORD *)v5 + 1) = v21;
        v22 = v80;
        v80 = 0;
        *((_QWORD *)v5 + 2) = v22;
        goto LABEL_32;
      }
      if ( (v9 & 0xFFFFFFFD) == 0 )
        goto LABEL_79;
      if ( HIDWORD(v75) != 2 )
        break;
      if ( !v70 )
        goto LABEL_79;
      a2 = v3;
      sub_A4D380(v81.m128i_i64, v3);
      v10 = v81.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v81.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v5[32] |= 3u;
        *(_QWORD *)v5 = v10;
        goto LABEL_32;
      }
      a2 = *(_QWORD *)(v3 + 48) - 16LL;
      sub_A4D330((__int64)(v70 + 1), (__int64 *)a2);
      v11 = *(_QWORD *)(v3 + 48);
      *(_QWORD *)(v3 + 48) = v11 - 16;
      v12 = *(volatile signed __int32 **)(v11 - 8);
      if ( v12 )
      {
        if ( &_pthread_key_create )
        {
          v13 = _InterlockedExchangeAdd(v12 + 2, 0xFFFFFFFF);
        }
        else
        {
          v13 = *((_DWORD *)v12 + 2);
          v7 = (unsigned int)(v13 - 1);
          *((_DWORD *)v12 + 2) = v7;
        }
        if ( v13 == 1 )
        {
          (*(void (**)(void))(*(_QWORD *)v12 + 16LL))();
          if ( &_pthread_key_create )
          {
            v56 = _InterlockedExchangeAdd(v12 + 3, 0xFFFFFFFF);
          }
          else
          {
            v56 = *((_DWORD *)v12 + 3);
            *((_DWORD *)v12 + 3) = v56 - 1;
          }
          if ( v56 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v12 + 24LL))(v12);
        }
      }
      if ( (v74 & 2) != 0 )
        sub_9CEF10(&v73);
      if ( (v74 & 1) != 0 )
      {
        v14 = v73;
        if ( v73 )
LABEL_18:
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v14 + 8LL))(v14);
      }
    }
    a2 = v3;
    LODWORD(v84) = 0;
    sub_A4B600((__int64)&v76, v3, SHIDWORD(v75), (__int64)v4, 0);
    v42 = v77 & 1;
    v7 = (unsigned int)(2 * v42);
    v43 = (2 * v42) | v77 & 0xFD;
    v77 = v43;
    if ( (_BYTE)v42 )
      break;
    switch ( (_DWORD)v76 )
    {
      case 2:
        if ( !v70 )
        {
LABEL_79:
          v41 = v5[32];
          v5[24] = 0;
          v5[32] = v41 & 0xFC | 2;
          goto LABEL_32;
        }
        if ( a3 )
        {
          a2 = v83;
          v81.m128i_i64[0] = (__int64)&v82;
          sub_A4AAA0(&v81, (char *)v83, (char *)(v83 + 8LL * (unsigned int)v84));
          v54 = (__int64 *)v70[4];
          if ( (__m128i *)v81.m128i_i64[0] == &v82 )
          {
            v59 = v81.m128i_i64[1];
            if ( v81.m128i_i64[1] )
            {
              if ( v81.m128i_i64[1] == 1 )
              {
                *(_BYTE *)v54 = v82.m128i_i8[0];
              }
              else
              {
                a2 = (__int64)&v82;
                memcpy(v54, &v82, v81.m128i_u64[1]);
              }
              v59 = v81.m128i_i64[1];
              v54 = (__int64 *)v70[4];
            }
            v70[5] = v59;
            *((_BYTE *)v54 + v59) = 0;
          }
          else
          {
            v55 = v82.m128i_i64[0];
            v7 = v81.m128i_i64[1];
            if ( v54 == v70 + 6 )
            {
              a2 = (__int64)v70;
              v70[4] = v81.m128i_i64[0];
              v70[5] = v7;
              v70[6] = v55;
            }
            else
            {
              a2 = v70[6];
              v70[4] = v81.m128i_i64[0];
              v70[5] = v7;
              v70[6] = v55;
              if ( v54 )
              {
                v81.m128i_i64[0] = (__int64)v54;
                v82.m128i_i64[0] = a2;
                goto LABEL_141;
              }
            }
            v81.m128i_i64[0] = (__int64)&v82;
          }
LABEL_141:
          v81.m128i_i64[1] = 0;
          *(_BYTE *)v81.m128i_i64[0] = 0;
          v52 = v81.m128i_i64[0];
          if ( (__m128i *)v81.m128i_i64[0] == &v82 )
          {
LABEL_132:
            v43 = v77;
            v53 = (v77 & 2) != 0;
            goto LABEL_133;
          }
LABEL_131:
          a2 = v82.m128i_i64[0] + 1;
          j_j___libc_free_0(v52, v82.m128i_i64[0] + 1);
          goto LABEL_132;
        }
        break;
      case 3:
        if ( !v70 )
          goto LABEL_79;
        if ( a3 )
        {
          v81.m128i_i64[0] = (__int64)&v82;
          sub_A4AAA0(&v81, (char *)(v83 + 8), (char *)(v83 + 8LL * (unsigned int)v84));
          LODWORD(v75) = *(_QWORD *)v83;
          a2 = v70[9];
          if ( a2 == v70[10] )
          {
            sub_A4B020(v70 + 8, (const __m128i *)a2, &v75, &v81);
          }
          else
          {
            if ( a2 )
            {
              *(_DWORD *)a2 = v75;
              *(_QWORD *)(a2 + 8) = a2 + 24;
              if ( (__m128i *)v81.m128i_i64[0] == &v82 )
              {
                *(__m128i *)(a2 + 24) = _mm_load_si128(&v82);
              }
              else
              {
                *(_QWORD *)(a2 + 8) = v81.m128i_i64[0];
                *(_QWORD *)(a2 + 24) = v82.m128i_i64[0];
              }
              *(_QWORD *)(a2 + 16) = v81.m128i_i64[1];
              v81.m128i_i64[0] = (__int64)&v82;
              v81.m128i_i64[1] = 0;
              v82.m128i_i8[0] = 0;
            }
            v70[9] += 40;
          }
          v52 = v81.m128i_i64[0];
          if ( (__m128i *)v81.m128i_i64[0] == &v82 )
            goto LABEL_132;
          goto LABEL_131;
        }
        break;
      case 1:
        if ( !(_DWORD)v84 )
          goto LABEL_79;
        a2 = (__int64)v79;
        v50 = *(_QWORD *)v83;
        v51 = v78;
        v7 = (unsigned int)*(_QWORD *)v83;
        if ( v79 == v78 )
        {
          if ( v79 != v80 )
          {
            if ( v79 )
              goto LABEL_151;
LABEL_155:
            v79 = (const __m128i *)((char *)v79 + 88);
LABEL_156:
            v79[-6].m128i_i32[2] = v50;
            v70 = &v79[-6].m128i_i64[1];
            v43 = v77;
            v53 = (v77 & 2) != 0;
LABEL_133:
            if ( v53 )
              sub_9CE230(&v76);
            goto LABEL_86;
          }
LABEL_158:
          v72 = *(_QWORD *)v83;
          sub_A4ABB0((__int64 *)&v78, v79);
          LODWORD(v50) = v72;
          goto LABEL_156;
        }
        if ( (_DWORD)v50 != v79[-6].m128i_i32[2] )
        {
          do
          {
            if ( (_DWORD)v7 == v51->m128i_i32[0] )
            {
              v70 = (__int64 *)v51;
              goto LABEL_3;
            }
            v51 = (const __m128i *)((char *)v51 + 88);
          }
          while ( v79 != v51 );
          if ( v79 != v80 )
          {
LABEL_151:
            v7 = 22;
            v57 = v79;
            while ( v7 )
            {
              v57->m128i_i32[0] = 0;
              v57 = (const __m128i *)((char *)v57 + 4);
              --v7;
            }
            *(_QWORD *)(a2 + 32) = a2 + 48;
            goto LABEL_155;
          }
          goto LABEL_158;
        }
        v70 = &v79[-6].m128i_i64[1];
        break;
      default:
LABEL_86:
        if ( (v43 & 1) != 0 )
        {
          v14 = v76;
          if ( v76 )
            goto LABEL_18;
        }
        break;
    }
  }
  v5[32] |= 3u;
  v77 = v43 & 0xFD;
  v58 = v76;
  v76 = 0;
  *(_QWORD *)v5 = v58 & 0xFFFFFFFFFFFFFFFELL;
LABEL_32:
  if ( (_BYTE *)v83 != v85 )
    _libc_free(v83, a2);
  v23 = v78;
  v71 = v79;
  if ( v79 != v78 )
  {
    v69 = v5;
    do
    {
      v24 = v23[4].m128i_i64[1];
      v25 = v23[4].m128i_i64[0];
      if ( v24 != v25 )
      {
        do
        {
          v26 = *(_QWORD *)(v25 + 8);
          if ( v26 != v25 + 24 )
            j_j___libc_free_0(v26, *(_QWORD *)(v25 + 24) + 1LL);
          v25 += 40;
        }
        while ( v24 != v25 );
        v25 = v23[4].m128i_i64[0];
      }
      if ( v25 )
        j_j___libc_free_0(v25, v23[5].m128i_i64[0] - v25);
      v27 = (const __m128i *)v23[2].m128i_i64[0];
      if ( v27 != &v23[3] )
        j_j___libc_free_0(v27, v23[3].m128i_i64[0] + 1);
      v28 = v23[1].m128i_i64[0];
      v29 = v23->m128i_i64[1];
      if ( v28 != v29 )
      {
        do
        {
          while ( 1 )
          {
            v30 = *(volatile signed __int32 **)(v29 + 8);
            if ( v30 )
            {
              if ( &_pthread_key_create )
              {
                v31 = _InterlockedExchangeAdd(v30 + 2, 0xFFFFFFFF);
              }
              else
              {
                v31 = *((_DWORD *)v30 + 2);
                *((_DWORD *)v30 + 2) = v31 - 1;
              }
              if ( v31 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v30 + 16LL))(v30);
                if ( &_pthread_key_create )
                {
                  v32 = _InterlockedExchangeAdd(v30 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v32 = *((_DWORD *)v30 + 3);
                  *((_DWORD *)v30 + 3) = v32 - 1;
                }
                if ( v32 == 1 )
                  break;
              }
            }
            v29 += 16;
            if ( v28 == v29 )
              goto LABEL_56;
          }
          v29 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v30 + 24LL))(v30);
        }
        while ( v28 != v29 );
LABEL_56:
        v29 = v23->m128i_i64[1];
      }
      if ( v29 )
        j_j___libc_free_0(v29, v23[1].m128i_i64[1] - v29);
      v23 = (const __m128i *)((char *)v23 + 88);
    }
    while ( v71 != v23 );
    v5 = v69;
    v23 = v78;
  }
  if ( v23 )
    j_j___libc_free_0(v23, (char *)v80 - (char *)v23);
  return v5;
}
