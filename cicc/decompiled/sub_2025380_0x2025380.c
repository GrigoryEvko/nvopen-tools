// Function: sub_2025380
// Address: 0x2025380
//
void __fastcall sub_2025380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6, __m128i a7)
{
  unsigned __int64 *v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // r15
  __int64 v12; // rcx
  unsigned int v13; // r13d
  unsigned __int64 v14; // rcx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int8 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // r11
  unsigned int *v23; // r10
  int v24; // eax
  __int64 v25; // rdx
  char v26; // cl
  __int64 v27; // rax
  __int8 v28; // dl
  unsigned __int64 v29; // rax
  unsigned int v30; // r11d
  char *v31; // rax
  __int64 v32; // rsi
  char v33; // dl
  __int64 v34; // rax
  __m128i v35; // xmm0
  unsigned int v36; // r11d
  __m128i v37; // kr00_16
  int v38; // r8d
  __int64 v39; // r9
  __m128i *p_s; // rdi
  __int64 v41; // r9
  __int64 v42; // rdx
  int v43; // ecx
  __int64 v44; // rax
  _QWORD *v45; // r13
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 *v49; // rdi
  int v50; // edx
  int v51; // edx
  __int64 v52; // rax
  __m128i *v53; // rdi
  int v54; // edx
  int v55; // eax
  _QWORD *v56; // [rsp+0h] [rbp-150h]
  int v57; // [rsp+0h] [rbp-150h]
  __int64 v58; // [rsp+8h] [rbp-148h]
  unsigned int v59; // [rsp+14h] [rbp-13Ch]
  __int64 v60; // [rsp+18h] [rbp-138h]
  __int64 v61; // [rsp+20h] [rbp-130h]
  __int64 *v62; // [rsp+20h] [rbp-130h]
  unsigned int v65; // [rsp+3Ch] [rbp-114h]
  __int64 v66; // [rsp+40h] [rbp-110h]
  unsigned int v67; // [rsp+40h] [rbp-110h]
  int v68; // [rsp+40h] [rbp-110h]
  const void *v69; // [rsp+40h] [rbp-110h]
  __int64 v70; // [rsp+40h] [rbp-110h]
  __int64 v71; // [rsp+48h] [rbp-108h]
  __int64 v72; // [rsp+80h] [rbp-D0h] BYREF
  int v73; // [rsp+88h] [rbp-C8h]
  __int128 v74; // [rsp+90h] [rbp-C0h] BYREF
  __int128 v75; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+B0h] [rbp-A0h] BYREF
  int v77; // [rsp+B8h] [rbp-98h]
  __m128 v78; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v79; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v80; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v81; // [rsp+E8h] [rbp-68h]
  __m128i v82; // [rsp+F0h] [rbp-60h] BYREF
  __m128i s; // [rsp+100h] [rbp-50h] BYREF

  v65 = *(unsigned __int16 *)(a2 + 24);
  v9 = *(unsigned __int64 **)(a2 + 32);
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *v9;
  v12 = v9[1];
  v72 = v10;
  v13 = *((_DWORD *)v9 + 2);
  v66 = v12;
  v14 = v11;
  if ( v10 )
  {
    sub_1623A60((__int64)&v72, v10, 2);
    v14 = v11;
  }
  v15 = *(_DWORD *)(a2 + 64);
  v16 = *(_QWORD *)(a1 + 8);
  DWORD2(v74) = 0;
  DWORD2(v75) = 0;
  v17 = *(_QWORD *)a1;
  v73 = v15;
  v18 = *(_QWORD *)(v16 + 48);
  v19 = (unsigned __int8 *)(*(_QWORD *)(v14 + 40) + 16LL * v13);
  *(_QWORD *)&v74 = 0;
  v20 = *((_QWORD *)v19 + 1);
  *(_QWORD *)&v75 = 0;
  sub_1F40D10((__int64)&v82, v17, v18, *v19, v20);
  if ( v82.m128i_i8[0] == 6 )
  {
    sub_2017DE0(a1, v11, v66, &v74, &v75);
  }
  else
  {
    v21 = *(_QWORD *)(a2 + 72);
    v22 = *(_QWORD *)(a1 + 8);
    v76 = v21;
    if ( v21 )
    {
      v61 = v22;
      sub_1623A60((__int64)&v76, v21, 2);
      v22 = v61;
    }
    v23 = *(unsigned int **)(a2 + 32);
    v24 = *(_DWORD *)(a2 + 64);
    v79.m128i_i8[0] = 0;
    v78.m128_i8[0] = 0;
    v77 = v24;
    v78.m128_u64[1] = 0;
    v79.m128i_i64[1] = 0;
    v60 = (__int64)v23;
    v62 = (__int64 *)v22;
    v25 = *(_QWORD *)(*(_QWORD *)v23 + 40LL) + 16LL * v23[2];
    v26 = *(_BYTE *)v25;
    v81 = *(_QWORD *)(v25 + 8);
    LOBYTE(v80) = v26;
    sub_1D19A30((__int64)&v82, v22, &v80);
    a6 = _mm_loadu_si128(&v82);
    a7 = _mm_loadu_si128(&s);
    v78 = (__m128)a6;
    v79 = a7;
    sub_1D40600(
      (__int64)&v82,
      v62,
      v60,
      (__int64)&v76,
      (const void ***)&v78,
      (const void ***)&v79,
      a5,
      *(double *)a6.m128i_i64,
      a7);
    if ( v76 )
      sub_161E7C0((__int64)&v76, v76);
    *(_QWORD *)&v74 = v82.m128i_i64[0];
    DWORD2(v74) = v82.m128i_i32[2];
    *(_QWORD *)&v75 = s.m128i_i64[0];
    DWORD2(v75) = s.m128i_i32[2];
  }
  v27 = *(_QWORD *)(v74 + 40) + 16LL * DWORD2(v74);
  v28 = *(_BYTE *)v27;
  v29 = *(_QWORD *)(v27 + 8);
  v78.m128_i8[0] = v28;
  v78.m128_u64[1] = v29;
  if ( v28 )
    v30 = word_4305480[(unsigned __int8)(v28 - 14)];
  else
    v30 = sub_1F58D30((__int64)&v78);
  v31 = *(char **)(a2 + 40);
  v32 = *(_QWORD *)(a1 + 8);
  v79.m128i_i8[0] = 0;
  v79.m128i_i64[1] = 0;
  v33 = *v31;
  v34 = *((_QWORD *)v31 + 1);
  v67 = v30;
  LOBYTE(v80) = v33;
  v81 = v34;
  sub_1D19A30((__int64)&v82, v32, &v80);
  v35 = _mm_loadu_si128(&v82);
  v36 = v67;
  v79 = v35;
  v37 = s;
  if ( v82.m128i_i8[0] )
  {
    v38 = word_4305480[(unsigned __int8)(v82.m128i_i8[0] - 14)];
  }
  else
  {
    v55 = sub_1F58D30((__int64)&v79);
    v36 = v67;
    v38 = v55;
  }
  v39 = v36;
  v82.m128i_i64[0] = (__int64)&s;
  p_s = &s;
  v82.m128i_i64[1] = 0x800000000LL;
  if ( v36 > 8 )
  {
    v59 = v36;
    v57 = v38;
    v70 = v36;
    sub_16CD150((__int64)&v82, &s, v36, 4, v38, v36);
    p_s = (__m128i *)v82.m128i_i64[0];
    v36 = v59;
    v38 = v57;
    v39 = v70;
  }
  v41 = 4 * v39;
  v82.m128i_i32[2] = v36;
  v42 = (__int64)p_s->m128i_i64 + v41;
  if ( p_s != (__m128i *)&p_s->m128i_i8[v41] )
  {
    v68 = v38;
    memset(p_s, 255, v41);
    v42 = v82.m128i_i64[0];
    v38 = v68;
  }
  if ( v38 )
  {
    v43 = 2 * v38;
    v44 = 0;
    do
    {
      *(_DWORD *)(v42 + v44) = v38++;
      v42 = v82.m128i_i64[0];
      v44 += 4;
    }
    while ( v38 != v43 );
  }
  v45 = *(_QWORD **)(a1 + 8);
  v69 = (const void *)v42;
  v80 = 0;
  v71 = v82.m128i_u32[2];
  LODWORD(v81) = 0;
  v46 = sub_1D2B300(v45, 0x30u, (__int64)&v80, v78.m128_u32[0], v78.m128_i64[1], v41);
  if ( v80 )
  {
    v56 = v46;
    v58 = v47;
    sub_161E7C0((__int64)&v80, v80);
    v46 = v56;
    v47 = v58;
  }
  v48 = sub_1D41320(
          (__int64)v45,
          v78.m128_u32[0],
          (const void **)v78.m128_u64[1],
          (__int64)&v72,
          v74,
          *((__int64 *)&v74 + 1),
          *(double *)v35.m128i_i64,
          *(double *)a6.m128i_i64,
          a7,
          (__int64)v46,
          v47,
          v69,
          v71);
  v49 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v75 = v48;
  DWORD2(v75) = v50;
  *(_QWORD *)a3 = sub_1D309E0(
                    v49,
                    v65,
                    (__int64)&v72,
                    v79.m128i_u32[0],
                    (const void **)v79.m128i_i64[1],
                    0,
                    *(double *)v35.m128i_i64,
                    *(double *)a6.m128i_i64,
                    *(double *)a7.m128i_i64,
                    v74);
  *(_DWORD *)(a3 + 8) = v51;
  v52 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          v65,
          (__int64)&v72,
          v37.m128i_i64[0],
          (const void **)v37.m128i_i64[1],
          0,
          *(double *)v35.m128i_i64,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128i_i64,
          v75);
  v53 = (__m128i *)v82.m128i_i64[0];
  *(_QWORD *)a4 = v52;
  *(_DWORD *)(a4 + 8) = v54;
  if ( v53 != &s )
    _libc_free((unsigned __int64)v53);
  if ( v72 )
    sub_161E7C0((__int64)&v72, v72);
}
