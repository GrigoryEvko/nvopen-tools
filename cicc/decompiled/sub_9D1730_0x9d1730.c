// Function: sub_9D1730
// Address: 0x9d1730
//
__m128i *__fastcall sub_9D1730(__m128i *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __m128i *v5; // r13
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rbx
  unsigned int *v11; // r14
  char v12; // al
  const char *v13; // rax
  __int64 v14; // rax
  __int32 v16; // eax
  __int32 v17; // eax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // r15
  __int64 v24; // rdi
  __int64 v25; // r14
  __int64 v26; // rbx
  volatile signed __int32 *v27; // r12
  signed __int32 v28; // eax
  void (*v29)(); // rax
  signed __int32 v30; // eax
  __int64 (__fastcall *v31)(__int64); // rdx
  __int64 v32; // rbx
  __int64 v33; // r12
  volatile signed __int32 *v34; // r14
  signed __int32 v35; // eax
  void (*v36)(); // rax
  signed __int32 v37; // eax
  __int64 (__fastcall *v38)(__int64); // rdx
  __int64 v39; // r12
  unsigned __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rax
  _BYTE *v44; // [rsp+18h] [rbp-208h]
  __m128i v45; // [rsp+20h] [rbp-200h]
  __m128i v46; // [rsp+30h] [rbp-1F0h]
  __int64 v47; // [rsp+58h] [rbp-1C8h] BYREF
  __int64 v48; // [rsp+60h] [rbp-1C0h] BYREF
  char v49; // [rsp+68h] [rbp-1B8h]
  __int64 v50[2]; // [rsp+70h] [rbp-1B0h] BYREF
  _QWORD v51[2]; // [rsp+80h] [rbp-1A0h] BYREF
  __m128i v52; // [rsp+90h] [rbp-190h] BYREF
  __m128i v53; // [rsp+A0h] [rbp-180h] BYREF
  __int64 v54; // [rsp+B0h] [rbp-170h]
  __int64 v55; // [rsp+B8h] [rbp-168h]
  __int64 v56; // [rsp+C0h] [rbp-160h]
  __int64 v57; // [rsp+C8h] [rbp-158h]
  _BYTE *v58; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v59; // [rsp+D8h] [rbp-148h]
  _BYTE v60[256]; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v61; // [rsp+1E0h] [rbp-40h]

  v5 = a1;
  if ( (a3 & 3) != 0 )
  {
    BYTE1(v54) = 1;
    v13 = "Invalid bitcode signature";
    goto LABEL_15;
  }
  if ( a2 != a2 + a3 )
  {
    if ( *(_BYTE *)a2 != 0xDE || *(_BYTE *)(a2 + 1) != 0xC0 || *(_BYTE *)(a2 + 2) != 23 || *(_BYTE *)(a2 + 3) != 11 )
      goto LABEL_4;
    if ( (unsigned int)a3 > 0xF )
    {
      a4 = *(unsigned int *)(a2 + 8);
      v40 = *(unsigned int *)(a2 + 12);
      a1 = (__m128i *)(v40 + a4);
      if ( a3 >= v40 + a4 )
      {
        a2 += a4;
        a3 = v40;
        goto LABEL_4;
      }
    }
    BYTE1(v54) = 1;
    v13 = "Invalid bitcode wrapper header";
LABEL_15:
    v52.m128i_i64[0] = (__int64)v13;
    LOBYTE(v54) = 3;
    sub_9C8190(v50, (__int64)&v52);
    v14 = v50[0];
    v5[21].m128i_i8[8] |= 3u;
    v5->m128i_i64[0] = v14 & 0xFFFFFFFFFFFFFFFELL;
    return v5;
  }
  a3 = 0;
LABEL_4:
  v52.m128i_i64[0] = a2;
  v54 = 0x200000000LL;
  v58 = v60;
  v52.m128i_i64[1] = a3;
  v53 = 0u;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v59 = 0x800000000LL;
  v61 = 0;
  if ( a3 <= 3 )
  {
    v39 = sub_2241E50(a1, a2, a3, a4, a5);
    v50[0] = (__int64)v51;
    sub_9C2D70(v50, "file too small to contain bitcode header", (__int64)"");
    v9 = (__int64)v50;
    sub_C63F00(&v47, v50, 84, v39);
    if ( (_QWORD *)v50[0] != v51 )
    {
      v9 = v51[0] + 1LL;
      j_j___libc_free_0(v50[0], v51[0] + 1LL);
    }
    goto LABEL_65;
  }
  sub_9C66D0((__int64)&v48, (__int64)&v52, 8, a4);
  v8 = v49 & 1;
  v9 = (unsigned int)(2 * v8);
  v49 = (2 * v8) | v49 & 0xFD;
  if ( (_BYTE)v8 )
    goto LABEL_83;
  if ( v48 != 66 )
  {
LABEL_75:
    v41 = sub_2241E50(&v48, v9, v8, v6, v7);
    v50[0] = (__int64)v51;
    sub_9C2D70(v50, "file doesn't start with bitcode header", (__int64)"");
    v9 = (__int64)v50;
    sub_C63F00(&v47, v50, 84, v41);
    if ( (_QWORD *)v50[0] != v51 )
    {
      v9 = v51[0] + 1LL;
      j_j___libc_free_0(v50[0], v51[0] + 1LL);
    }
    if ( (v49 & 2) != 0 )
      sub_9CDF70(&v48);
    if ( (v49 & 1) != 0 && v48 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v48 + 8LL))(v48);
    goto LABEL_65;
  }
  sub_9C66D0((__int64)&v48, (__int64)&v52, 8, v6);
  v8 = v49 & 1;
  v9 = (unsigned int)(2 * v8);
  v49 = (2 * v8) | v49 & 0xFD;
  if ( (_BYTE)v8 )
  {
LABEL_83:
    v43 = v48;
    v49 &= ~2u;
    v48 = 0;
    v47 = v43 | 1;
    goto LABEL_65;
  }
  v10 = 0;
  v11 = (unsigned int *)&unk_3F222B0;
  if ( v48 != 67 )
    goto LABEL_75;
  while ( 1 )
  {
    sub_9C66D0((__int64)&v48, (__int64)&v52, 4, v6);
    v8 = v49 & 1;
    v9 = (unsigned int)(2 * v8);
    v12 = (2 * v8) | v49 & 0xFD;
    v49 = v12;
    if ( (_BYTE)v8 )
      break;
    if ( v48 != v10 )
      goto LABEL_75;
    if ( ++v11 == (unsigned int *)&unk_3F222C0 )
      goto LABEL_17;
    v10 = *v11;
  }
  v49 = v12 & 0xFD;
  v42 = v48;
  v48 = 0;
  v47 = v42 | 1;
LABEL_65:
  if ( (v47 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v5->m128i_i64[0] = v47 & 0xFFFFFFFFFFFFFFFELL;
    v21 = (unsigned int)v59;
    v5[21].m128i_i8[8] |= 3u;
  }
  else
  {
LABEL_17:
    v45 = _mm_loadu_si128(&v52);
    v46 = _mm_loadu_si128(&v53);
    v5[21].m128i_i8[8] = v5[21].m128i_i8[8] & 0xFC | 2;
    v16 = v54;
    *v5 = v45;
    v5[2].m128i_i32[0] = v16;
    v17 = HIDWORD(v54);
    v5[1] = v46;
    v5[2].m128i_i32[1] = v17;
    v18 = v55;
    v55 = 0;
    v5[2].m128i_i64[1] = v18;
    v19 = v56;
    v56 = 0;
    v5[3].m128i_i64[0] = v19;
    v20 = v57;
    v57 = 0;
    v5[3].m128i_i64[1] = v20;
    v5[4].m128i_i64[0] = (__int64)v5[5].m128i_i64;
    v5[4].m128i_i64[1] = 0x800000000LL;
    v21 = (unsigned int)v59;
    if ( (_DWORD)v59 )
    {
      v9 = (__int64)&v58;
      sub_9D06B0((__int64)v5[4].m128i_i64, (__int64)&v58);
      v21 = (unsigned int)v59;
    }
    v5[21].m128i_i64[0] = v61;
  }
  v22 = 32 * v21;
  v23 = &v58[v22];
  v44 = v58;
  if ( v58 != &v58[v22] )
  {
    while ( 1 )
    {
      v24 = *((_QWORD *)v23 - 3);
      v25 = *((_QWORD *)v23 - 2);
      v23 -= 32;
      v26 = v24;
      if ( v25 != v24 )
        break;
LABEL_36:
      if ( v24 )
      {
        v9 = *((_QWORD *)v23 + 3) - v24;
        j_j___libc_free_0(v24, v9);
      }
      if ( v44 == v23 )
      {
        v23 = v58;
        goto LABEL_40;
      }
    }
    while ( 1 )
    {
      v27 = *(volatile signed __int32 **)(v26 + 8);
      if ( !v27 )
        goto LABEL_23;
      if ( &_pthread_key_create )
      {
        v28 = _InterlockedExchangeAdd(v27 + 2, 0xFFFFFFFF);
      }
      else
      {
        v28 = *((_DWORD *)v27 + 2);
        v9 = (unsigned int)(v28 - 1);
        *((_DWORD *)v27 + 2) = v9;
      }
      if ( v28 != 1 )
        goto LABEL_23;
      v29 = *(void (**)())(*(_QWORD *)v27 + 16LL);
      if ( v29 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v29)(v27);
      if ( &_pthread_key_create )
      {
        v30 = _InterlockedExchangeAdd(v27 + 3, 0xFFFFFFFF);
      }
      else
      {
        v30 = *((_DWORD *)v27 + 3);
        *((_DWORD *)v27 + 3) = v30 - 1;
      }
      if ( v30 != 1 )
        goto LABEL_23;
      v31 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 24LL);
      if ( v31 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 8LL))(v27);
        v26 += 16;
        if ( v25 == v26 )
        {
LABEL_35:
          v24 = *((_QWORD *)v23 + 1);
          goto LABEL_36;
        }
      }
      else
      {
        v31((__int64)v27);
LABEL_23:
        v26 += 16;
        if ( v25 == v26 )
          goto LABEL_35;
      }
    }
  }
LABEL_40:
  if ( v23 != v60 )
    _libc_free(v23, v9);
  v32 = v56;
  v33 = v55;
  if ( v56 != v55 )
  {
    while ( 1 )
    {
      v34 = *(volatile signed __int32 **)(v33 + 8);
      if ( !v34 )
        goto LABEL_44;
      if ( &_pthread_key_create )
      {
        v35 = _InterlockedExchangeAdd(v34 + 2, 0xFFFFFFFF);
      }
      else
      {
        v35 = *((_DWORD *)v34 + 2);
        *((_DWORD *)v34 + 2) = v35 - 1;
      }
      if ( v35 != 1 )
        goto LABEL_44;
      v36 = *(void (**)())(*(_QWORD *)v34 + 16LL);
      if ( v36 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v36)(v34);
      if ( &_pthread_key_create )
      {
        v37 = _InterlockedExchangeAdd(v34 + 3, 0xFFFFFFFF);
      }
      else
      {
        v37 = *((_DWORD *)v34 + 3);
        *((_DWORD *)v34 + 3) = v37 - 1;
      }
      if ( v37 != 1 )
        goto LABEL_44;
      v38 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v34 + 24LL);
      if ( v38 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v34 + 8LL))(v34);
        v33 += 16;
        if ( v32 == v33 )
        {
LABEL_56:
          v33 = v55;
          break;
        }
      }
      else
      {
        v38((__int64)v34);
LABEL_44:
        v33 += 16;
        if ( v32 == v33 )
          goto LABEL_56;
      }
    }
  }
  if ( v33 )
    j_j___libc_free_0(v33, v57 - v33);
  return v5;
}
