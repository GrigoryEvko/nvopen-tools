// Function: sub_A11D10
// Address: 0xa11d10
//
_BYTE *__fastcall sub_A11D10(_BYTE *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r12
  char v5; // al
  const __m128i *v7; // r13
  __int32 v9; // eax
  __int32 v10; // eax
  unsigned __int64 v11; // rbx
  char *v12; // rax
  char *v13; // rsi
  char *v14; // rdx
  char *v15; // rsi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  int v20; // edx
  char v21; // al
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  _BYTE *v24; // rbx
  _BYTE *v25; // r12
  __int64 v26; // rdi
  __int64 v27; // r15
  __int64 v28; // r13
  volatile signed __int32 *v29; // r14
  signed __int32 v30; // eax
  void (*v31)(); // rax
  signed __int32 v32; // eax
  char *v33; // rbx
  char *v34; // r13
  volatile signed __int32 *v35; // r14
  signed __int32 v36; // eax
  void (*v37)(); // rax
  signed __int32 v38; // eax
  __int64 (__fastcall *v39)(__int64); // rdx
  unsigned __int64 v40; // rax
  char v41; // al
  __int64 v42; // rax
  _QWORD *v43; // rdx
  _BYTE *v44; // rdx
  __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // [rsp+8h] [rbp-448h]
  unsigned __int64 v50; // [rsp+8h] [rbp-448h]
  unsigned int v51; // [rsp+14h] [rbp-43Ch]
  int v52; // [rsp+18h] [rbp-438h]
  _BYTE *v54; // [rsp+28h] [rbp-428h]
  __int64 v55; // [rsp+68h] [rbp-3E8h] BYREF
  unsigned __int64 v56; // [rsp+70h] [rbp-3E0h] BYREF
  char v57; // [rsp+78h] [rbp-3D8h]
  unsigned __int64 v58; // [rsp+80h] [rbp-3D0h] BYREF
  char v59; // [rsp+88h] [rbp-3C8h]
  char v60; // [rsp+A0h] [rbp-3B0h]
  char v61; // [rsp+A1h] [rbp-3AFh]
  __m128i v62; // [rsp+B0h] [rbp-3A0h] BYREF
  __m128i v63; // [rsp+C0h] [rbp-390h]
  unsigned __int32 v64; // [rsp+D0h] [rbp-380h]
  __int32 v65; // [rsp+D4h] [rbp-37Ch]
  char *v66; // [rsp+D8h] [rbp-378h]
  char *v67; // [rsp+E0h] [rbp-370h]
  char *v68; // [rsp+E8h] [rbp-368h]
  _BYTE *v69; // [rsp+F0h] [rbp-360h] BYREF
  __int64 v70; // [rsp+F8h] [rbp-358h]
  _BYTE v71[256]; // [rsp+100h] [rbp-350h] BYREF
  __int64 v72; // [rsp+200h] [rbp-250h]
  _BYTE *v73; // [rsp+210h] [rbp-240h] BYREF
  __int64 v74; // [rsp+218h] [rbp-238h]
  _BYTE v75[560]; // [rsp+220h] [rbp-230h] BYREF

  v4 = a1;
  if ( !a2[47].m128i_i64[1] )
  {
    v5 = a1[8];
    *a1 = 1;
    a1[8] = v5 & 0xFC | 2;
    return v4;
  }
  v7 = (const __m128i *)a2[15].m128i_i64[0];
  v9 = v7[2].m128i_i32[0];
  v63 = _mm_loadu_si128(v7 + 1);
  v64 = v9;
  v10 = v7[2].m128i_i32[1];
  v11 = v7[3].m128i_i64[0] - v7[2].m128i_i64[1];
  v62 = _mm_loadu_si128(v7);
  v65 = v10;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  if ( v11 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4261EA(a1, a2, a3, a4);
    v12 = (char *)sub_22077B0(v11);
  }
  else
  {
    v12 = 0;
  }
  v67 = v12;
  v68 = &v12[v11];
  v13 = (char *)v7[3].m128i_i64[0];
  v66 = v12;
  v14 = (char *)v7[2].m128i_i64[1];
  if ( v13 == v14 )
  {
    v15 = v12;
  }
  else
  {
    v15 = &v12[v13 - v14];
    do
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = *(_QWORD *)v14;
        a4 = *((_QWORD *)v14 + 1);
        *((_QWORD *)v12 + 1) = a4;
        if ( a4 )
        {
          if ( &_pthread_key_create )
            _InterlockedAdd((volatile signed __int32 *)(a4 + 8), 1u);
          else
            ++*(_DWORD *)(a4 + 8);
        }
      }
      v12 += 16;
      v14 += 16;
    }
    while ( v12 != v15 );
  }
  v67 = v15;
  v69 = v71;
  v70 = 0x800000000LL;
  if ( v7[4].m128i_i32[2] )
    sub_A05260((__int64)&v69, (__int64)v7[4].m128i_i64, (__int64)v14);
  v16 = a2[47].m128i_u64[1];
  v17 = (unsigned __int64)&v62;
  v72 = v7[21].m128i_i64[0];
  v74 = 0x4000000000LL;
  v73 = v75;
  sub_9CDFE0((__int64 *)&v58, (__int64)&v62, v16, a4);
  v19 = v58 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v58 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    while ( 1 )
    {
      v17 = (unsigned __int64)&v62;
      sub_9CEFB0((__int64)&v58, (__int64)&v62, 1, v18);
      if ( (v59 & 1) != 0 )
      {
        v59 &= ~2u;
        v40 = v58;
        v58 = 0;
        v56 = v40 | 1;
      }
      else
      {
        v56 = 1;
        v51 = HIDWORD(v58);
        v52 = v58;
      }
      v19 = v56 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v56 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( v52 == 1 )
        goto LABEL_82;
      if ( (v52 & 0xFFFFFFFD) == 0 )
      {
        v17 = (unsigned __int64)&v58;
        v61 = 1;
        v58 = (unsigned __int64)"Malformed block";
        v60 = 3;
        sub_A01DB0((__int64 *)&v56, (__int64)&v58);
        v42 = v56;
        a1[8] |= 3u;
        *(_QWORD *)a1 = v42 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_32;
      }
      v17 = (unsigned __int64)&v62;
      v49 = 8 * v63.m128i_i64[0] - v64;
      sub_A4CAE0(&v56, &v62, v51);
      v20 = v57 & 1;
      v21 = (2 * v20) | v57 & 0xFD;
      v57 = v21;
      if ( (_BYTE)v20 )
      {
        a1[8] |= 3u;
        v57 = v21 & 0xFD;
        v47 = v56;
        v56 = 0;
        *(_QWORD *)a1 = v47 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_32;
      }
      if ( (_DWORD)v56 != 36 )
      {
LABEL_82:
        v41 = a1[8];
        *a1 = 1;
        a1[8] = v41 & 0xFC | 2;
        goto LABEL_32;
      }
      v17 = (unsigned __int64)&v62;
      sub_9CDFE0((__int64 *)&v58, (__int64)&v62, v49, (unsigned int)(2 * v20));
      v22 = v58 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v58 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_25;
      v17 = (unsigned __int64)&v62;
      LODWORD(v74) = 0;
      sub_A4B600(&v58, &v62, v51, &v73, 0);
      if ( (v59 & 1) != 0 )
      {
        v48 = v58;
        a1[8] |= 3u;
        *(_QWORD *)a1 = v48 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_26;
      }
      v18 = (unsigned int)v74;
      if ( (v74 & 1) == 0
        || (v43 = (_QWORD *)a2[14].m128i_i64[1],
            (unsigned int)*(_QWORD *)v73 >= (unsigned int)((__int64)(v43[1] - *v43) >> 5)) )
      {
        v17 = (unsigned __int64)&v58;
        v61 = 1;
        v58 = (unsigned __int64)"Invalid record";
        v60 = 3;
        sub_A01DB0(&v55, (__int64)&v58);
        a1[8] |= 3u;
        *(_QWORD *)a1 = v55 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_26;
      }
      v44 = *(_BYTE **)(*v43 + 32LL * (unsigned int)*(_QWORD *)v73 + 16);
      if ( (unsigned __int8)(*v44 - 2) <= 1u || !*v44 )
      {
        v17 = (unsigned __int64)a2;
        v50 = 8 * v63.m128i_i64[0] - v64;
        sub_A119B0((__int64 *)&v58, a2, (__int64)v44, (__int64)(v73 + 8), v74 - 1);
        v46 = v58 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v58 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          a1[8] |= 3u;
          *(_QWORD *)a1 = v46;
          goto LABEL_26;
        }
        v17 = (unsigned __int64)&v62;
        sub_9CDFE0((__int64 *)&v58, (__int64)&v62, v50, v45);
        v22 = v58 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v58 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
LABEL_25:
          a1[8] |= 3u;
          *(_QWORD *)a1 = v22;
LABEL_26:
          if ( (v57 & 2) != 0 )
LABEL_98:
            sub_9CE230(&v56);
          if ( (v57 & 1) != 0 && v56 )
            (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v56 + 8LL))(v56);
          goto LABEL_32;
        }
      }
      if ( (v57 & 2) != 0 )
        goto LABEL_98;
      if ( (v57 & 1) != 0 && v56 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v56 + 8LL))(v56);
    }
  }
  a1[8] |= 3u;
  *(_QWORD *)a1 = v19;
LABEL_32:
  if ( v73 != v75 )
    _libc_free(v73, v17);
  v23 = 32LL * (unsigned int)v70;
  v54 = v69;
  v24 = &v69[v23];
  if ( v69 != &v69[v23] )
  {
    v25 = &v69[v23];
    while ( 1 )
    {
      v26 = *((_QWORD *)v25 - 3);
      v27 = *((_QWORD *)v25 - 2);
      v25 -= 32;
      v28 = v26;
      if ( v27 != v26 )
        break;
LABEL_51:
      if ( v26 )
      {
        v17 = *((_QWORD *)v25 + 3) - v26;
        j_j___libc_free_0(v26, v17);
      }
      if ( v54 == v25 )
      {
        v4 = a1;
        v24 = v69;
        goto LABEL_55;
      }
    }
    while ( 1 )
    {
      v29 = *(volatile signed __int32 **)(v28 + 8);
      if ( !v29 )
        goto LABEL_38;
      if ( &_pthread_key_create )
      {
        v30 = _InterlockedExchangeAdd(v29 + 2, 0xFFFFFFFF);
      }
      else
      {
        v30 = *((_DWORD *)v29 + 2);
        v17 = (unsigned int)(v30 - 1);
        *((_DWORD *)v29 + 2) = v17;
      }
      if ( v30 != 1 )
        goto LABEL_38;
      v31 = *(void (**)())(*(_QWORD *)v29 + 16LL);
      if ( v31 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v31)(v29);
      if ( &_pthread_key_create )
      {
        v32 = _InterlockedExchangeAdd(v29 + 3, 0xFFFFFFFF);
      }
      else
      {
        v32 = *((_DWORD *)v29 + 3);
        v17 = (unsigned int)(v32 - 1);
        *((_DWORD *)v29 + 3) = v17;
      }
      if ( v32 != 1 )
        goto LABEL_38;
      v17 = *(_QWORD *)(*(_QWORD *)v29 + 24LL);
      if ( (__int64 (__fastcall *)(__int64))v17 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v29 + 8LL))(v29);
        v28 += 16;
        if ( v27 == v28 )
        {
LABEL_50:
          v26 = *((_QWORD *)v25 + 1);
          goto LABEL_51;
        }
      }
      else
      {
        ((void (__fastcall *)(volatile signed __int32 *))v17)(v29);
LABEL_38:
        v28 += 16;
        if ( v27 == v28 )
          goto LABEL_50;
      }
    }
  }
LABEL_55:
  if ( v24 != v71 )
    _libc_free(v24, v17);
  v33 = v67;
  v34 = v66;
  if ( v67 != v66 )
  {
    do
    {
      v35 = (volatile signed __int32 *)*((_QWORD *)v34 + 1);
      if ( v35 )
      {
        if ( &_pthread_key_create )
        {
          v36 = _InterlockedExchangeAdd(v35 + 2, 0xFFFFFFFF);
        }
        else
        {
          v36 = *((_DWORD *)v35 + 2);
          *((_DWORD *)v35 + 2) = v36 - 1;
        }
        if ( v36 == 1 )
        {
          v37 = *(void (**)())(*(_QWORD *)v35 + 16LL);
          if ( v37 != nullsub_25 )
            ((void (__fastcall *)(volatile signed __int32 *))v37)(v35);
          if ( &_pthread_key_create )
          {
            v38 = _InterlockedExchangeAdd(v35 + 3, 0xFFFFFFFF);
          }
          else
          {
            v38 = *((_DWORD *)v35 + 3);
            *((_DWORD *)v35 + 3) = v38 - 1;
          }
          if ( v38 == 1 )
          {
            v39 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v35 + 24LL);
            if ( v39 == sub_9C26E0 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v35 + 8LL))(v35);
            else
              v39((__int64)v35);
          }
        }
      }
      v34 += 16;
    }
    while ( v33 != v34 );
    v34 = v66;
  }
  if ( v34 )
    j_j___libc_free_0(v34, v68 - v34);
  return v4;
}
