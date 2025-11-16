// Function: sub_22E39A0
// Address: 0x22e39a0
//
void __fastcall sub_22E39A0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  unsigned __int64 *v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int8 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rax
  unsigned __int64 *v12; // rsi
  __int64 *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  const __m128i *v16; // rcx
  const __m128i *v17; // rdx
  unsigned __int64 v18; // rbx
  __m128i *v19; // rax
  __int64 v20; // rcx
  const __m128i *v21; // rax
  const __m128i *v22; // rcx
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __m128i *v26; // rdx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // r9
  unsigned __int64 v32; // r14
  __int64 v33; // rbx
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  int v36; // edx
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  int v39; // eax
  unsigned int v40; // esi
  __int64 v41; // rdi
  unsigned __int64 *v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r15
  unsigned __int64 *v47; // rax
  unsigned __int64 v48; // rax
  char v49; // si
  char v50; // dl
  __m128i *v51; // rax
  __m128i si128; // xmm0
  __int64 v53; // [rsp+8h] [rbp-258h]
  __m128i v55; // [rsp+20h] [rbp-240h] BYREF
  char v56; // [rsp+38h] [rbp-228h]
  __int64 v57; // [rsp+40h] [rbp-220h] BYREF
  unsigned __int64 *v58; // [rsp+48h] [rbp-218h]
  __int64 v59; // [rsp+50h] [rbp-210h]
  int v60; // [rsp+58h] [rbp-208h]
  char v61; // [rsp+5Ch] [rbp-204h]
  unsigned __int64 v62; // [rsp+60h] [rbp-200h] BYREF
  _QWORD v63[7]; // [rsp+68h] [rbp-1F8h] BYREF
  __m128i *v64; // [rsp+A0h] [rbp-1C0h] BYREF
  unsigned __int64 v65; // [rsp+A8h] [rbp-1B8h]
  __int8 *v66; // [rsp+B0h] [rbp-1B0h]
  unsigned __int64 v67[16]; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v68; // [rsp+140h] [rbp-120h] BYREF
  char v69; // [rsp+158h] [rbp-108h]
  char v70; // [rsp+15Ch] [rbp-104h]
  char v71[64]; // [rsp+160h] [rbp-100h] BYREF
  const __m128i *v72; // [rsp+1A0h] [rbp-C0h]
  const __m128i *v73; // [rsp+1A8h] [rbp-B8h]
  __int8 *v74; // [rsp+1B0h] [rbp-B0h]
  char v75[8]; // [rsp+1B8h] [rbp-A8h] BYREF
  unsigned __int64 v76; // [rsp+1C0h] [rbp-A0h]
  char v77; // [rsp+1D4h] [rbp-8Ch]
  char v78[64]; // [rsp+1D8h] [rbp-88h] BYREF
  const __m128i *v79; // [rsp+218h] [rbp-48h]
  unsigned __int64 v80; // [rsp+220h] [rbp-40h]
  unsigned __int64 v81; // [rsp+228h] [rbp-38h]

  sub_CB6200(*(_QWORD *)(a1 + 208), *(unsigned __int8 **)(a1 + 176), *(_QWORD *)(a1 + 184));
  v2 = a2[4];
  v58 = &v62;
  v53 = v2;
  memset(v67, 0, 0x78u);
  LODWORD(v67[2]) = 8;
  v67[1] = (unsigned __int64)&v67[4];
  v3 = *a2;
  BYTE4(v67[3]) = 1;
  v64 = 0;
  v62 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v68.m128i_i64[0] = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v59 = 0x100000008LL;
  v65 = 0;
  v66 = 0;
  v60 = 0;
  v61 = 1;
  v57 = 1;
  v69 = 0;
  sub_22E3960((__int64)&v64, &v68);
  v4 = &v62;
  if ( &v62 == v63 )
  {
LABEL_88:
    ++HIDWORD(v59);
    v63[0] = v53;
    ++v57;
  }
  else
  {
    while ( v53 != *v4 )
    {
      if ( v63 == ++v4 )
        goto LABEL_88;
    }
  }
  sub_C8CF70((__int64)&v68, v71, 8, (__int64)&v62, (__int64)&v57);
  v5 = (unsigned __int64)v64;
  v64 = 0;
  v72 = (const __m128i *)v5;
  v6 = v65;
  v65 = 0;
  v73 = (const __m128i *)v6;
  v7 = v66;
  v66 = 0;
  v74 = v7;
  sub_C8CF70((__int64)v75, v78, 8, (__int64)&v67[4], (__int64)v67);
  v11 = v67[12];
  memset(&v67[12], 0, 24);
  v79 = (const __m128i *)v11;
  v80 = v67[13];
  v81 = v67[14];
  if ( v64 )
    j_j___libc_free_0((unsigned __int64)v64);
  if ( !v61 )
    _libc_free((unsigned __int64)v58);
  if ( v67[12] )
    j_j___libc_free_0(v67[12]);
  if ( !BYTE4(v67[3]) )
    _libc_free(v67[1]);
  v12 = &v62;
  v13 = &v57;
  sub_C8CD80((__int64)&v57, (__int64)&v62, (__int64)&v68, v8, v9, v10);
  v16 = v73;
  v17 = v72;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v18 = (char *)v73 - (char *)v72;
  if ( v73 == v72 )
  {
    v19 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_92;
    v19 = (__m128i *)sub_22077B0((char *)v73 - (char *)v72);
    v16 = v73;
    v17 = v72;
  }
  v64 = v19;
  v65 = (unsigned __int64)v19;
  v66 = &v19->m128i_i8[v18];
  if ( v17 == v16 )
  {
    v20 = (__int64)v19;
  }
  else
  {
    v20 = (__int64)v19->m128i_i64 + (char *)v16 - (char *)v17;
    do
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v17);
        v19[1] = _mm_loadu_si128(v17 + 1);
      }
      v19 += 2;
      v17 += 2;
    }
    while ( (__m128i *)v20 != v19 );
  }
  v12 = &v67[4];
  v13 = (__int64 *)v67;
  v65 = v20;
  sub_C8CD80((__int64)v67, (__int64)&v67[4], (__int64)v75, v20, v14, v15);
  v21 = (const __m128i *)v80;
  v22 = v79;
  memset(&v67[12], 0, 24);
  v23 = v80 - (_QWORD)v79;
  if ( (const __m128i *)v80 != v79 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v24 = sub_22077B0(v80 - (_QWORD)v79);
      v22 = v79;
      v25 = v24;
      v21 = (const __m128i *)v80;
      goto LABEL_26;
    }
LABEL_92:
    sub_4261EA(v13, v12, v17);
  }
  v25 = 0;
LABEL_26:
  v67[12] = v25;
  v67[13] = v25;
  v67[14] = v25 + v23;
  if ( v21 == v22 )
  {
    v27 = v25;
  }
  else
  {
    v26 = (__m128i *)v25;
    v27 = v25 + (char *)v21 - (char *)v22;
    do
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(v22);
        v26[1] = _mm_loadu_si128(v22 + 1);
      }
      v26 += 2;
      v22 += 2;
    }
    while ( (__m128i *)v27 != v26 );
  }
  v28 = v65;
  v29 = (unsigned __int64)v64;
  v67[13] = v27;
  if ( v65 - (_QWORD)v64 != v27 - v25 )
    goto LABEL_32;
  while ( v29 != v28 )
  {
    v48 = v25;
    while ( *(_QWORD *)v29 == *(_QWORD *)v48 )
    {
      v49 = *(_BYTE *)(v29 + 24);
      if ( v49 != *(_BYTE *)(v48 + 24) || v49 && *(_DWORD *)(v29 + 16) != *(_DWORD *)(v48 + 16) )
        break;
      v29 += 32LL;
      v48 += 32LL;
      if ( v28 == v29 )
        goto LABEL_60;
    }
    do
    {
LABEL_32:
      v30 = *(_QWORD *)(v28 - 32);
      v31 = *(_QWORD *)(a1 + 208);
      if ( v30 )
      {
        sub_A68DD0(v30, *(_QWORD *)(a1 + 208), 0, 0, 0);
      }
      else
      {
        v51 = *(__m128i **)(v31 + 32);
        if ( *(_QWORD *)(v31 + 24) - (_QWORD)v51 <= 0x14u )
        {
          sub_CB6200(*(_QWORD *)(a1 + 208), "Printing <null> Block", 0x15u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
          v51[1].m128i_i32[0] = 1668246594;
          v51[1].m128i_i8[4] = 107;
          *v51 = si128;
          *(_QWORD *)(v31 + 32) += 21LL;
        }
      }
      v32 = v65;
      do
      {
        v33 = *(_QWORD *)(v32 - 32);
        v34 = v33 + 48;
        if ( !*(_BYTE *)(v32 - 8) )
        {
          v35 = *(_QWORD *)(v33 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v34 == v35 )
          {
            v37 = 0;
          }
          else
          {
            if ( !v35 )
              BUG();
            v36 = *(unsigned __int8 *)(v35 - 24);
            v37 = v35 - 24;
            if ( (unsigned int)(v36 - 30) >= 0xB )
              v37 = 0;
          }
          *(_QWORD *)(v32 - 24) = v37;
          *(_DWORD *)(v32 - 16) = 0;
          *(_BYTE *)(v32 - 8) = 1;
        }
LABEL_41:
        v38 = *(_QWORD *)(v33 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v34 == v38 )
          goto LABEL_78;
LABEL_42:
        if ( !v38 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v38 - 24) - 30 <= 0xA )
        {
          v39 = sub_B46E30(v38 - 24);
          v40 = *(_DWORD *)(v32 - 16);
          if ( v40 == v39 )
            goto LABEL_79;
          goto LABEL_45;
        }
LABEL_78:
        while ( 1 )
        {
          v40 = *(_DWORD *)(v32 - 16);
          if ( !v40 )
            break;
LABEL_45:
          v41 = *(_QWORD *)(v32 - 24);
          *(_DWORD *)(v32 - 16) = v40 + 1;
          v46 = sub_B46EC0(v41, v40);
          if ( v61 )
          {
            v47 = v58;
            v42 = &v58[HIDWORD(v59)];
            if ( v58 != v42 )
            {
              while ( v46 != *v47 )
              {
                if ( v42 == ++v47 )
                  goto LABEL_49;
              }
              goto LABEL_41;
            }
LABEL_49:
            if ( HIDWORD(v59) < (unsigned int)v59 )
            {
              ++HIDWORD(v59);
              *v42 = v46;
              ++v57;
LABEL_51:
              v55.m128i_i64[0] = v46;
              v56 = 0;
              sub_22E3960((__int64)&v64, &v55);
              v29 = (unsigned __int64)v64;
              v28 = v65;
              goto LABEL_52;
            }
          }
          sub_C8CC70((__int64)&v57, v46, (__int64)v42, v43, v44, v45);
          if ( v50 )
            goto LABEL_51;
          v38 = *(_QWORD *)(v33 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v34 != v38 )
            goto LABEL_42;
        }
LABEL_79:
        v65 -= 32LL;
        v29 = (unsigned __int64)v64;
        v32 = v65;
      }
      while ( (__m128i *)v65 != v64 );
      v28 = (unsigned __int64)v64;
LABEL_52:
      v25 = v67[12];
    }
    while ( v28 - v29 != v67[13] - v67[12] );
  }
LABEL_60:
  if ( v25 )
    j_j___libc_free_0(v25);
  if ( !BYTE4(v67[3]) )
    _libc_free(v67[1]);
  if ( v64 )
    j_j___libc_free_0((unsigned __int64)v64);
  if ( !v61 )
    _libc_free((unsigned __int64)v58);
  if ( v79 )
    j_j___libc_free_0((unsigned __int64)v79);
  if ( !v77 )
    _libc_free(v76);
  if ( v72 )
    j_j___libc_free_0((unsigned __int64)v72);
  if ( !v70 )
    _libc_free(v68.m128i_u64[1]);
}
