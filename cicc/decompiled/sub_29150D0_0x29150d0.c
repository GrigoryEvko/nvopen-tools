// Function: sub_29150D0
// Address: 0x29150d0
//
__int64 __fastcall sub_29150D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // r14
  __int64 v7; // r13
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  _BYTE *v11; // r8
  __int64 v12; // r9
  unsigned __int8 **v13; // rbx
  unsigned __int8 **v14; // r15
  char **v15; // r11
  unsigned __int8 v16; // al
  unsigned __int8 **v17; // rdi
  bool v19; // al
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __m128i *v23; // rbx
  __int64 v24; // r12
  __int64 v25; // r14
  __int64 i; // r13
  unsigned __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rax
  int v30; // edx
  char *v31; // rax
  __int64 v32; // rax
  char *v33; // rax
  __m128i v34; // xmm0
  __int64 v35; // [rsp+8h] [rbp-F8h]
  unsigned __int8 **v36; // [rsp+10h] [rbp-F0h]
  char **v37; // [rsp+20h] [rbp-E0h]
  __int64 v38; // [rsp+28h] [rbp-D8h]
  char *v39; // [rsp+28h] [rbp-D8h]
  __m128i v40; // [rsp+30h] [rbp-D0h] BYREF
  char **v41; // [rsp+40h] [rbp-C0h]
  char v42; // [rsp+4Fh] [rbp-B1h]
  __m128i v43; // [rsp+50h] [rbp-B0h] BYREF
  char *v44; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+68h] [rbp-98h]
  _BYTE v46[32]; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int8 **v47; // [rsp+90h] [rbp-70h] BYREF
  int v48; // [rsp+98h] [rbp-68h]
  _BYTE v49[96]; // [rsp+A0h] [rbp-60h] BYREF

  v6 = v46;
  v7 = a2;
  v45 = 0x200000000LL;
  v42 = a3;
  v44 = v46;
  sub_2914D90((__int64)&v47, a2, a3, a4, a5, a6);
  v13 = v47;
  v14 = &v47[v48];
  if ( v47 == v14 )
  {
LABEL_31:
    if ( v14 != (unsigned __int8 **)v49 )
      _libc_free((unsigned __int64)v14);
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x200000000LL;
    if ( (_DWORD)v45 )
      sub_29138A0(a1, &v44, v9, v10, (__int64)v11, v12);
    *(_BYTE *)(a1 + 48) = 1;
    goto LABEL_7;
  }
  v15 = &v44;
  while ( 1 )
  {
    v11 = *v13;
    v16 = **v13;
    if ( v16 == 78 )
    {
      v32 = *((_QWORD *)v11 + 2);
      if ( !v32 || *(_QWORD *)(v32 + 8) )
        goto LABEL_5;
      v11 = *(_BYTE **)(v32 + 24);
      v16 = *v11;
    }
    if ( v16 <= 0x1Cu )
      goto LABEL_5;
    if ( v16 != 62 )
      break;
    if ( (v11[2] & 1) != 0 || v42 )
      goto LABEL_5;
    v29 = (unsigned int)v45;
    v10 = HIDWORD(v45);
    v30 = v45;
    if ( (unsigned int)v45 >= (unsigned __int64)HIDWORD(v45) )
    {
      v9 = (unsigned int)v45 + 1LL;
      v43.m128i_i64[0] = (__int64)v11;
      v43.m128i_i8[8] = 1;
      v34 = _mm_load_si128(&v43);
      if ( HIDWORD(v45) < v9 )
      {
        v41 = v15;
        v40 = v34;
        sub_C8D5F0((__int64)v15, v6, v9, 0x10u, (__int64)v11, v12);
        v29 = (unsigned int)v45;
        v34 = _mm_load_si128(&v40);
        v15 = v41;
      }
      goto LABEL_47;
    }
    v31 = &v44[16 * (unsigned int)v45];
    if ( v31 )
    {
      *(_QWORD *)v31 = v11;
      v31[8] = 1;
      v30 = v45;
    }
LABEL_28:
    v9 = (unsigned int)(v30 + 1);
    LODWORD(v45) = v9;
LABEL_29:
    if ( v14 == ++v13 )
    {
      v14 = v47;
      goto LABEL_31;
    }
  }
  v41 = v15;
  if ( v16 != 61 || (v11[2] & 1) != 0 )
    goto LABEL_5;
  v40.m128i_i64[0] = (__int64)v11;
  v19 = sub_B46500(v11);
  v11 = (_BYTE *)v40.m128i_i64[0];
  v15 = v41;
  if ( v19 )
  {
    if ( v42 )
      goto LABEL_5;
LABEL_40:
    v10 = HIDWORD(v45);
    v30 = v45;
    if ( (unsigned int)v45 >= (unsigned __int64)HIDWORD(v45) )
    {
      v9 = (unsigned int)v45 + 1LL;
      v43.m128i_i64[0] = (__int64)v11;
      v43.m128i_i8[8] = 0;
      v34 = _mm_load_si128(&v43);
      if ( HIDWORD(v45) < v9 )
      {
        v41 = v15;
        v40 = v34;
        sub_C8D5F0((__int64)v15, v6, v9, 0x10u, (__int64)v11, v12);
        v34 = _mm_load_si128(&v40);
        v15 = v41;
      }
      v29 = (unsigned int)v45;
LABEL_47:
      *(__m128i *)&v44[16 * v29] = v34;
      LODWORD(v45) = v45 + 1;
      goto LABEL_29;
    }
    v33 = &v44[16 * (unsigned int)v45];
    if ( v33 )
    {
      *(_QWORD *)v33 = v11;
      v33[8] = 0;
      v30 = v45;
    }
    goto LABEL_28;
  }
  v37 = v41;
  v38 = v40.m128i_i64[0];
  v20 = sub_B43CC0(v7);
  v21 = *(_QWORD *)(v7 - 64);
  LOBYTE(v41) = 0;
  v40.m128i_i64[0] = v20;
  v22 = *(_QWORD *)(v7 - 32);
  v43.m128i_i64[0] = v21;
  v43.m128i_i64[1] = v22;
  v36 = v13;
  v23 = &v43;
  v35 = a1;
  v24 = v38;
  v39 = v6;
  v25 = v7;
  for ( i = v21; ; i = v23->m128i_i64[0] )
  {
    _BitScanReverse64(&v27, 1LL << (*(_WORD *)(v24 + 2) >> 1));
    if ( (unsigned __int8)sub_D31180(
                            i,
                            *(_QWORD *)(v24 + 8),
                            63 - ((unsigned int)v27 ^ 0x3F),
                            v40.m128i_i64[0],
                            v24,
                            0,
                            0,
                            0) )
    {
      LOBYTE(v41) = *(_QWORD *)(v25 - 64) == i ? (unsigned __int8)v41 | 1 : (unsigned __int8)v41 | 2;
    }
    else if ( v42 )
    {
      v28 = v24;
      v7 = v25;
      v15 = v37;
      v13 = v36;
      a1 = v35;
      v6 = v39;
      goto LABEL_20;
    }
    v23 = (__m128i *)((char *)v23 + 8);
    if ( v37 == (char **)v23 )
      break;
  }
  v28 = v24;
  v7 = v25;
  v15 = v37;
  v13 = v36;
  a1 = v35;
  v6 = v39;
  if ( !v42 )
  {
LABEL_22:
    v11 = (_BYTE *)((2LL * (unsigned __int8)v41) | v28 & 0xFFFFFFFFFFFFFFF9LL);
    goto LABEL_40;
  }
LABEL_20:
  if ( ((unsigned __int8)v41 & 1) != 0 && ((unsigned __int8)v41 & 2) != 0 )
    goto LABEL_22;
LABEL_5:
  v17 = v47;
  *(_QWORD *)(a1 + 48) = 0;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
  *(_OWORD *)(a1 + 32) = 0;
  if ( v17 != (unsigned __int8 **)v49 )
    _libc_free((unsigned __int64)v17);
LABEL_7:
  if ( v44 != v6 )
    _libc_free((unsigned __int64)v44);
  return a1;
}
