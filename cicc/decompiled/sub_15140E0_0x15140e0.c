// Function: sub_15140E0
// Address: 0x15140e0
//
__int64 __fastcall sub_15140E0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rbx
  __int8 *v6; // r14
  __int8 *v7; // r15
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __m128i *v10; // r14
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rdi
  __m128i *v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // r12
  volatile signed __int32 *v17; // r13
  signed __int32 v18; // eax
  signed __int32 v19; // eax
  int v20; // eax
  __m128i *v21; // rsi
  __int64 v22; // r12
  __m128i *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __m128i *v26; // rsi
  __m128i *v27; // rax
  __m128i *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rax
  volatile signed __int32 *v34; // r12
  signed __int32 v35; // eax
  signed __int32 v36; // eax
  __int8 *v37; // rax
  char v38; // r10
  unsigned __int64 v39; // r12
  unsigned int v40; // ebx
  unsigned __int64 v41; // r13
  __int8 *v42; // r15
  int v43; // r14d
  unsigned __int64 v44; // rax
  size_t **v45; // r9
  unsigned __int64 v46; // rbx
  __int8 v47; // r14
  __int64 v48; // r13
  __m128i *v49; // r12
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  const void *v52; // r9
  size_t v53; // r8
  __m128i *v54; // rax
  __int64 v55; // rsi
  __m128i *v56; // rdi
  __int64 v57; // rax
  __m128i *v58; // rdi
  size_t **v59; // [rsp+0h] [rbp-300h]
  char v60; // [rsp+8h] [rbp-2F8h]
  __int8 *v61; // [rsp+10h] [rbp-2F0h]
  char v62; // [rsp+10h] [rbp-2F0h]
  char n; // [rsp+18h] [rbp-2E8h]
  size_t na; // [rsp+18h] [rbp-2E8h]
  size_t nb; // [rsp+18h] [rbp-2E8h]
  void *src; // [rsp+20h] [rbp-2E0h]
  __int8 *srca; // [rsp+20h] [rbp-2E0h]
  void *srcb; // [rsp+20h] [rbp-2E0h]
  __int64 v69; // [rsp+28h] [rbp-2D8h]
  __m128i *v71; // [rsp+38h] [rbp-2C8h]
  __int64 v72; // [rsp+48h] [rbp-2B8h] BYREF
  __m128i *v73; // [rsp+50h] [rbp-2B0h] BYREF
  __m128i *v74; // [rsp+58h] [rbp-2A8h]
  __m128i *v75; // [rsp+60h] [rbp-2A0h]
  _QWORD *v76; // [rsp+70h] [rbp-290h] BYREF
  unsigned __int64 v77; // [rsp+78h] [rbp-288h]
  _QWORD v78[2]; // [rsp+80h] [rbp-280h] BYREF
  size_t *v79; // [rsp+90h] [rbp-270h] BYREF
  __m128i *v80; // [rsp+98h] [rbp-268h] BYREF
  size_t v81; // [rsp+A0h] [rbp-260h] BYREF
  __m128i v82; // [rsp+A8h] [rbp-258h] BYREF
  __int8 *v83; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v84; // [rsp+C8h] [rbp-238h]
  _BYTE v85[560]; // [rsp+D0h] [rbp-230h] BYREF

  v4 = a2;
  if ( sub_15127D0(a2, 0, 0) )
  {
    *(_BYTE *)(a1 + 24) = 0;
    return a1;
  }
  v6 = 0;
  v7 = v85;
  v73 = 0;
  v84 = 0x4000000000LL;
  v74 = 0;
  v75 = 0;
  v83 = v85;
  while ( 1 )
  {
    v8 = sub_14ED070(v4, 2);
    if ( (_DWORD)v8 == 1 )
      break;
    if ( (v8 & 0xFFFFFFFD) == 0 )
      goto LABEL_9;
    v9 = HIDWORD(v8);
    if ( (_DWORD)v9 == 2 )
    {
      if ( !v6 )
        goto LABEL_9;
      sub_1513230(v4);
      v29 = *(_QWORD *)(v4 + 48);
      v30 = *((_QWORD *)v6 + 2);
      if ( v30 == *((_QWORD *)v6 + 3) )
      {
        sub_1512F90((char **)v6 + 1, (char *)v30, (__int64 *)(v29 - 16));
      }
      else
      {
        if ( v30 )
        {
          v31 = *(_QWORD *)(v29 - 16);
          *(_QWORD *)(v30 + 8) = 0;
          *(_QWORD *)(v29 - 16) = 0;
          *(_QWORD *)v30 = v31;
          v32 = *(_QWORD *)(v29 - 8);
          *(_QWORD *)(v29 - 8) = 0;
          *(_QWORD *)(v30 + 8) = v32;
          v30 = *((_QWORD *)v6 + 2);
        }
        *((_QWORD *)v6 + 2) = v30 + 16;
      }
      v33 = *(_QWORD *)(v4 + 48);
      *(_QWORD *)(v4 + 48) = v33 - 16;
      v34 = *(volatile signed __int32 **)(v33 - 8);
      if ( v34 )
      {
        if ( &_pthread_key_create )
        {
          v35 = _InterlockedExchangeAdd(v34 + 2, 0xFFFFFFFF);
        }
        else
        {
          v35 = *((_DWORD *)v34 + 2);
          *((_DWORD *)v34 + 2) = v35 - 1;
        }
        if ( v35 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v34 + 16LL))(v34);
          if ( &_pthread_key_create )
          {
            v36 = _InterlockedExchangeAdd(v34 + 3, 0xFFFFFFFF);
          }
          else
          {
            v36 = *((_DWORD *)v34 + 3);
            *((_DWORD *)v34 + 3) = v36 - 1;
          }
          if ( v36 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v34 + 24LL))(v34);
        }
      }
    }
    else
    {
      LODWORD(v84) = 0;
      v20 = sub_1510D70(v4, v9, (__int64)&v83, 0);
      switch ( v20 )
      {
        case 2:
          if ( !v6 )
          {
LABEL_9:
            *(_BYTE *)(a1 + 24) = 0;
            goto LABEL_10;
          }
          if ( a3 )
          {
            v80 = 0;
            v79 = &v81;
            LOBYTE(v81) = 0;
            if ( (_DWORD)v84 )
            {
              v69 = 8LL * (unsigned int)v84;
              v45 = &v79;
              srca = v6;
              na = v4;
              v46 = 1;
              v62 = a3;
              v47 = *v83;
              v48 = 8;
              v49 = 0;
              while ( 1 )
              {
                v49->m128i_i8[(_QWORD)v79] = v47;
                v80 = (__m128i *)v46;
                *((_BYTE *)v79 + v46) = 0;
                if ( v48 == v69 )
                  break;
                v49 = v80;
                v47 = v83[v48];
                v46 = (unsigned __int64)v80->m128i_u64 + 1;
                if ( v79 == &v81 )
                  v50 = 15;
                else
                  v50 = v81;
                v48 += 8;
                if ( v50 < v46 )
                {
                  v59 = v45;
                  sub_2240BB0(v45, v80, 0, 0, 1);
                  v45 = v59;
                }
              }
              v6 = srca;
              v4 = na;
              a3 = v62;
            }
            else
            {
              v45 = &v79;
            }
            sub_2240AE0(v6 + 32, v45);
            if ( v79 != &v81 )
              j_j___libc_free_0(v79, v81 + 1);
          }
          break;
        case 3:
          if ( !v6 )
            goto LABEL_9;
          if ( a3 )
          {
            v77 = 0;
            v76 = v78;
            v37 = v83;
            LOBYTE(v78[0]) = 0;
            if ( (_DWORD)v84 != 1 )
            {
              v38 = v83[8];
              src = (void *)v4;
              v39 = 0;
              v40 = 1;
              n = a3;
              v41 = 1;
              v61 = v7;
              v42 = v6;
              v43 = v84;
              while ( 1 )
              {
                ++v40;
                *((_BYTE *)v76 + v39) = v38;
                v77 = v41;
                *((_BYTE *)v76 + v41) = 0;
                if ( v40 == v43 )
                  break;
                v39 = v77;
                v38 = v83[8 * v40];
                v41 = v77 + 1;
                if ( v76 == v78 )
                  v44 = 15;
                else
                  v44 = v78[0];
                if ( v41 > v44 )
                {
                  v60 = v83[8 * v40];
                  sub_2240BB0(&v76, v77, 0, 0, 1);
                  v38 = v60;
                }
              }
              v6 = v42;
              v4 = (__int64)src;
              a3 = n;
              v7 = v61;
              v37 = v83;
            }
            v51 = *(_QWORD *)v37;
            v52 = v76;
            v53 = v77;
            v80 = &v82;
            LODWORD(v79) = v51;
            if ( (_QWORD *)((char *)v76 + v77) && !v76 )
              sub_426248((__int64)"basic_string::_M_construct null not valid");
            v72 = v77;
            if ( v77 > 0xF )
            {
              nb = v77;
              srcb = v76;
              v57 = sub_22409D0(&v80, &v72, 0);
              v52 = srcb;
              v53 = nb;
              v80 = (__m128i *)v57;
              v58 = (__m128i *)v57;
              v82.m128i_i64[0] = v72;
            }
            else
            {
              if ( v77 == 1 )
              {
                v82.m128i_i8[0] = *(_BYTE *)v76;
                v54 = &v82;
                goto LABEL_98;
              }
              if ( !v77 )
              {
                v54 = &v82;
                goto LABEL_98;
              }
              v58 = &v82;
            }
            memcpy(v58, v52, v53);
            v53 = v72;
            v54 = v80;
LABEL_98:
            v81 = v53;
            v54->m128i_i8[v53] = 0;
            v55 = *((_QWORD *)v6 + 9);
            if ( v55 == *((_QWORD *)v6 + 10) )
            {
              sub_1513E40((__int64 *)v6 + 8, (const __m128i *)v55, (__int64)&v79);
              v56 = v80;
            }
            else
            {
              v56 = v80;
              if ( v55 )
              {
                *(_DWORD *)v55 = (_DWORD)v79;
                *(_QWORD *)(v55 + 8) = v55 + 24;
                if ( v80 == &v82 )
                {
                  *(__m128i *)(v55 + 24) = _mm_loadu_si128(&v82);
                }
                else
                {
                  *(_QWORD *)(v55 + 8) = v80;
                  *(_QWORD *)(v55 + 24) = v82.m128i_i64[0];
                }
                *(_QWORD *)(v55 + 16) = v81;
                v81 = 0;
                v82.m128i_i8[0] = 0;
                v80 = &v82;
                v56 = &v82;
              }
              *((_QWORD *)v6 + 9) += 40LL;
            }
            if ( v56 != &v82 )
              j_j___libc_free_0(v56, v82.m128i_i64[0] + 1);
            if ( v76 != v78 )
              j_j___libc_free_0(v76, v78[0] + 1LL);
          }
          break;
        case 1:
          if ( !(_DWORD)v84 )
            goto LABEL_9;
          v21 = v74;
          v22 = *(_QWORD *)v83;
          v23 = v73;
          if ( v74 == v73 || (v6 = &v74[-6].m128i_i8[8], (_DWORD)v22 != v74[-6].m128i_i32[2]) )
          {
            v24 = 0x2E8BA2E8BA2E8BA3LL * (((char *)v74 - (char *)v73) >> 3);
            if ( (_DWORD)v24 )
            {
              v25 = (__int64)&v73[5].m128i_i64[11 * (unsigned int)(v24 - 1) + 1];
              while ( 1 )
              {
                v6 = (__int8 *)v23;
                if ( (unsigned int)*(_QWORD *)v83 == v23->m128i_i32[0] )
                  break;
                v23 = (__m128i *)((char *)v23 + 88);
                if ( (__m128i *)v25 == v23 )
                  goto LABEL_50;
              }
            }
            else
            {
LABEL_50:
              if ( v74 == v75 )
              {
                sub_1512360((__int64 *)&v73, v74);
                v26 = v74;
              }
              else
              {
                if ( v74 )
                {
                  memset(v74, 0, 0x58u);
                  v21[2].m128i_i64[0] = (__int64)v21[3].m128i_i64;
                  v21 = v74;
                }
                v26 = (__m128i *)((char *)v21 + 88);
                v74 = v26;
              }
              v26[-6].m128i_i32[2] = v22;
              v6 = &v74[-6].m128i_i8[8];
            }
          }
          break;
      }
    }
  }
  v27 = v73;
  v73 = 0;
  *(_QWORD *)a1 = v27;
  v28 = v74;
  *(_BYTE *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 8) = v28;
  v74 = 0;
  *(_QWORD *)(a1 + 16) = v75;
  v75 = 0;
LABEL_10:
  if ( v83 != v7 )
    _libc_free((unsigned __int64)v83);
  v10 = v73;
  v71 = v74;
  if ( v74 != v73 )
  {
    do
    {
      v11 = v10[4].m128i_i64[1];
      v12 = v10[4].m128i_i64[0];
      if ( v11 != v12 )
      {
        do
        {
          v13 = *(_QWORD *)(v12 + 8);
          if ( v13 != v12 + 24 )
            j_j___libc_free_0(v13, *(_QWORD *)(v12 + 24) + 1LL);
          v12 += 40;
        }
        while ( v11 != v12 );
        v12 = v10[4].m128i_i64[0];
      }
      if ( v12 )
        j_j___libc_free_0(v12, v10[5].m128i_i64[0] - v12);
      v14 = (__m128i *)v10[2].m128i_i64[0];
      if ( v14 != &v10[3] )
        j_j___libc_free_0(v14, v10[3].m128i_i64[0] + 1);
      v15 = v10[1].m128i_i64[0];
      v16 = v10->m128i_i64[1];
      if ( v15 != v16 )
      {
        do
        {
          while ( 1 )
          {
            v17 = *(volatile signed __int32 **)(v16 + 8);
            if ( v17 )
            {
              if ( &_pthread_key_create )
              {
                v18 = _InterlockedExchangeAdd(v17 + 2, 0xFFFFFFFF);
              }
              else
              {
                v18 = *((_DWORD *)v17 + 2);
                *((_DWORD *)v17 + 2) = v18 - 1;
              }
              if ( v18 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 16LL))(v17);
                if ( &_pthread_key_create )
                {
                  v19 = _InterlockedExchangeAdd(v17 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v19 = *((_DWORD *)v17 + 3);
                  *((_DWORD *)v17 + 3) = v19 - 1;
                }
                if ( v19 == 1 )
                  break;
              }
            }
            v16 += 16;
            if ( v15 == v16 )
              goto LABEL_33;
          }
          v16 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 24LL))(v17);
        }
        while ( v15 != v16 );
LABEL_33:
        v16 = v10->m128i_i64[1];
      }
      if ( v16 )
        j_j___libc_free_0(v16, v10[1].m128i_i64[1] - v16);
      v10 = (__m128i *)((char *)v10 + 88);
    }
    while ( v71 != v10 );
    v10 = v73;
  }
  if ( v10 )
    j_j___libc_free_0(v10, (char *)v75 - (char *)v10);
  return a1;
}
