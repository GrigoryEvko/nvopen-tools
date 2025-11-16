// Function: sub_1A333C0
// Address: 0x1a333c0
//
char __fastcall sub_1A333C0(
        __int64 a1,
        unsigned __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rbx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rax
  _QWORD *v16; // rdi
  _QWORD *v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdx
  unsigned __int8 *v26; // rdi
  size_t v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 *v30; // rsi
  double v31; // xmm4_8
  double v32; // xmm5_8
  unsigned int v33; // eax
  size_t v35; // rdx
  _QWORD v36[2]; // [rsp+0h] [rbp-140h] BYREF
  __m128i v37; // [rsp+10h] [rbp-130h] BYREF
  __int16 v38; // [rsp+20h] [rbp-120h]
  __m128i v39; // [rsp+30h] [rbp-110h] BYREF
  char v40; // [rsp+40h] [rbp-100h]
  char v41; // [rsp+41h] [rbp-FFh]
  __m128i v42[2]; // [rsp+50h] [rbp-F0h] BYREF
  __m128i v43; // [rsp+70h] [rbp-D0h] BYREF
  __int16 v44; // [rsp+80h] [rbp-C0h]
  __m128i v45[2]; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v46; // [rsp+B0h] [rbp-90h] BYREF
  char v47; // [rsp+C0h] [rbp-80h]
  char v48; // [rsp+C1h] [rbp-7Fh]
  __m128i v49[2]; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int8 *v50; // [rsp+F0h] [rbp-50h] BYREF
  size_t n; // [rsp+F8h] [rbp-48h]
  _QWORD src[8]; // [rsp+100h] [rbp-40h] BYREF

  v11 = a1 + 112;
  v12 = *a2;
  *(_QWORD *)(a1 + 112) = *a2;
  v13 = a2[1];
  *(_QWORD *)(a1 + 120) = v13;
  *(_BYTE *)(a1 + 152) = (__int64)a2[2] >> 2;
  *(_BYTE *)(a1 + 152) &= 1u;
  v14 = *(_QWORD *)(a1 + 56);
  v15 = *(_QWORD *)(a1 + 64);
  if ( v14 <= v12 )
  {
    *(_BYTE *)(a1 + 153) = v15 < v13;
  }
  else
  {
    *(_BYTE *)(a1 + 153) = 1;
    v12 = v14;
  }
  *(_QWORD *)(a1 + 128) = v12;
  if ( v15 > v13 )
    v15 = v13;
  *(_QWORD *)(a1 + 136) = v15;
  *(_QWORD *)(a1 + 144) = v15 - v12;
  v16 = (_QWORD *)(a2[2] & 0xFFFFFFFFFFFFFFF8LL);
  *(_QWORD *)(a1 + 160) = v16;
  *(_QWORD *)(a1 + 168) = *v16;
  v17 = sub_1648700((__int64)v16);
  *(_QWORD *)(a1 + 200) = v17[5];
  *(_QWORD *)(a1 + 208) = v17 + 3;
  v18 = v17[6];
  v50 = (unsigned __int8 *)v18;
  if ( v18 )
  {
    sub_1623A60((__int64)&v50, v18, 2);
    v19 = *(_QWORD *)(a1 + 192);
    if ( !v19 )
      goto LABEL_8;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 192);
    if ( !v19 )
    {
      v21 = v17[6];
      v50 = (unsigned __int8 *)v21;
      if ( v21 )
        goto LABEL_11;
      goto LABEL_38;
    }
  }
  sub_161E7C0(a1 + 192, v19);
LABEL_8:
  v20 = v50;
  *(_QWORD *)(a1 + 192) = v50;
  if ( v20 )
    sub_1623210((__int64)&v50, v20, a1 + 192);
  v21 = v17[6];
  v50 = (unsigned __int8 *)v21;
  if ( v21 )
  {
LABEL_11:
    sub_1623A60((__int64)&v50, v21, 2);
    v22 = *(_QWORD *)(a1 + 192);
    if ( !v22 )
      goto LABEL_13;
    goto LABEL_12;
  }
LABEL_38:
  v22 = *(_QWORD *)(a1 + 192);
  if ( !v22 )
    goto LABEL_15;
LABEL_12:
  sub_161E7C0(a1 + 192, v22);
LABEL_13:
  v23 = v50;
  *(_QWORD *)(a1 + 192) = v50;
  if ( v23 )
    sub_1623210((__int64)&v50, v23, a1 + 192);
LABEL_15:
  v24 = *(_QWORD *)(a1 + 48);
  v43.m128i_i64[0] = v11;
  v48 = 1;
  v46.m128i_i64[0] = (__int64)".";
  v47 = 3;
  v44 = 267;
  v41 = 1;
  v39.m128i_i64[0] = (__int64)".";
  v40 = 3;
  v36[0] = sub_1649960(v24);
  v36[1] = v25;
  v37.m128i_i64[0] = (__int64)v36;
  v38 = 261;
  sub_14EC200(v42, &v37, &v39);
  sub_14EC200(v45, v42, &v43);
  sub_14EC200(v49, v45, &v46);
  sub_16E2FC0((__int64 *)&v50, (__int64)v49);
  v26 = *(unsigned __int8 **)(a1 + 256);
  if ( v50 == (unsigned __int8 *)src )
  {
    v35 = n;
    if ( n )
    {
      if ( n == 1 )
        *v26 = src[0];
      else
        memcpy(v26, src, n);
      v35 = n;
      v26 = *(unsigned __int8 **)(a1 + 256);
    }
    *(_QWORD *)(a1 + 264) = v35;
    v26[v35] = 0;
    v26 = v50;
    goto LABEL_19;
  }
  v27 = n;
  v28 = src[0];
  if ( v26 == (unsigned __int8 *)(a1 + 272) )
  {
    *(_QWORD *)(a1 + 256) = v50;
    *(_QWORD *)(a1 + 264) = v27;
    *(_QWORD *)(a1 + 272) = v28;
  }
  else
  {
    v29 = *(_QWORD *)(a1 + 272);
    *(_QWORD *)(a1 + 256) = v50;
    *(_QWORD *)(a1 + 264) = v27;
    *(_QWORD *)(a1 + 272) = v28;
    if ( v26 )
    {
      v50 = v26;
      src[0] = v29;
      goto LABEL_19;
    }
  }
  v50 = (unsigned __int8 *)src;
  v26 = (unsigned __int8 *)src;
LABEL_19:
  n = 0;
  *v26 = 0;
  if ( v50 != (unsigned __int8 *)src )
    j_j___libc_free_0(v50, src[0] + 1LL);
  v30 = sub_1648700(*(_QWORD *)(a1 + 160));
  switch ( *((_BYTE *)v30 + 16) )
  {
    case '6':
      return sub_1A2F700((_BYTE **)a1, v30, a3, a4, a5, a6, v31, v32, a9, a10);
    case '7':
      return sub_1A31B60((__int64 *)a1, (__int64)v30, *(double *)a3.m128_u64, a4, a5);
    case '8':
    case '9':
    case ':':
    case ';':
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'O':
      return sub_1A33110(a1, (__int64)v30);
    case 'M':
      return sub_1A327C0((__m128i *)a1, (__int64)v30);
    case 'N':
      v33 = *(_DWORD *)(*(v30 - 3) + 36);
      if ( v33 == 135 )
        return sub_1A30D10(a1, (__int64)v30, *(double *)a3.m128_u64, a4, a5);
      if ( v33 <= 0x87 )
      {
        if ( v33 != 133 )
          return sub_1A2F320(a1, (__int64)v30);
        return sub_1A30D10(a1, (__int64)v30, *(double *)a3.m128_u64, a4, a5);
      }
      else
      {
        if ( v33 != 137 )
          return sub_1A2F320(a1, (__int64)v30);
        return sub_1A2FFA0(a1, (__int64)v30, *(double *)a3.m128_u64, a4, a5);
      }
  }
}
