// Function: sub_3753C50
// Address: 0x3753c50
//
__int64 __fastcall sub_3753C50(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v6; // rax
  unsigned __int8 *v7; // rsi
  char *v8; // rdx
  __int64 v9; // rax
  char *v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  __int64 v13; // rcx
  char *v14; // rax
  __int64 v15; // r12
  __int64 v17; // rax
  __int64 v18; // rbx
  const __m128i *v19; // roff
  char v20; // dl
  __int64 v21; // r8
  int v22; // ecx
  int v23; // r15d
  unsigned int i; // eax
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rcx
  const __m128i *v33; // rdx
  __m128i *v34; // rax
  __int64 v35; // rax
  int v36; // r15d
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // r8
  unsigned __int64 v40; // rdx
  __int64 v41; // rax
  const __m128i *v42; // r15
  unsigned __int64 v43; // r8
  __m128i *v44; // rax
  __int64 v45; // rsi
  __int64 (__fastcall *v46)(__int64); // rax
  __int64 v47; // rax
  __int64 v48; // rcx
  unsigned int v49; // edx
  unsigned int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rdx
  signed __int64 v53; // r15
  unsigned __int64 v54; // rsi
  __int64 v55; // rdi
  signed __int64 v56; // r15
  unsigned __int64 v57; // r15
  __int64 v58; // [rsp+0h] [rbp-110h]
  __int64 v59; // [rsp+10h] [rbp-100h]
  __int64 v60; // [rsp+18h] [rbp-F8h]
  _QWORD *v61; // [rsp+20h] [rbp-F0h]
  __int64 v62; // [rsp+28h] [rbp-E8h]
  unsigned int v63; // [rsp+28h] [rbp-E8h]
  __int64 v64; // [rsp+28h] [rbp-E8h]
  __int64 v65; // [rsp+28h] [rbp-E8h]
  __int64 v66; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v67; // [rsp+48h] [rbp-C8h] BYREF
  __m128i v68; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+60h] [rbp-B0h]
  __int64 v70; // [rsp+70h] [rbp-A0h] BYREF
  int v71; // [rsp+78h] [rbp-98h]
  __int64 v72; // [rsp+80h] [rbp-90h]
  unsigned __int64 v73; // [rsp+88h] [rbp-88h]
  __int64 v74; // [rsp+90h] [rbp-80h]
  unsigned __int64 v75; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v76; // [rsp+A8h] [rbp-68h]
  _BYTE v77[96]; // [rsp+B0h] [rbp-60h] BYREF

  v60 = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD **)(a2 + 40);
  v7 = *(unsigned __int8 **)(a2 + 48);
  v61 = v6;
  v67 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v67, (__int64)v7, 1);
  v8 = *(char **)(a2 + 8);
  v59 = *(_QWORD *)(a1[2] + 8);
  v9 = 24LL * *(_QWORD *)a2;
  v10 = &v8[v9];
  v11 = v9 >> 3;
  v12 = 0xAAAAAAAAAAAAAAABLL * v11;
  v13 = (__int64)(0xAAAAAAAAAAAAAAABLL * v11) >> 2;
  if ( v13 <= 0 )
  {
    v54 = 0xAAAAAAAAAAAAAAABLL * v11;
    v14 = *(char **)(a2 + 8);
LABEL_97:
    if ( v54 != 2 )
    {
      if ( v54 != 3 )
      {
        if ( v54 != 1 )
          goto LABEL_100;
LABEL_114:
        if ( *(_DWORD *)v14 != 2 )
          goto LABEL_100;
        goto LABEL_115;
      }
      if ( *(_DWORD *)v14 == 2 )
      {
LABEL_115:
        if ( v10 != v14 )
          goto LABEL_11;
LABEL_100:
        if ( v13 > 0 )
          goto LABEL_20;
LABEL_101:
        if ( v12 != 2 )
        {
          if ( v12 != 3 )
          {
            if ( v12 != 1 )
              goto LABEL_11;
            goto LABEL_104;
          }
          if ( *(_DWORD *)v8 != 1 )
            goto LABEL_21;
          v8 += 24;
        }
        if ( *(_DWORD *)v8 != 1 )
          goto LABEL_21;
        v8 += 24;
LABEL_104:
        if ( *(_DWORD *)v8 != 1 )
          goto LABEL_21;
        goto LABEL_11;
      }
      v14 += 24;
    }
    if ( *(_DWORD *)v14 != 2 )
    {
      v14 += 24;
      goto LABEL_114;
    }
    goto LABEL_115;
  }
  v14 = *(char **)(a2 + 8);
  while ( 1 )
  {
    if ( *(_DWORD *)v14 == 2 )
    {
      if ( v10 == v14 )
        goto LABEL_20;
      goto LABEL_11;
    }
    if ( *((_DWORD *)v14 + 6) == 2 )
    {
      if ( v10 == v14 + 24 )
        goto LABEL_20;
      goto LABEL_11;
    }
    if ( *((_DWORD *)v14 + 12) == 2 )
    {
      if ( v10 == v14 + 48 )
        goto LABEL_20;
      goto LABEL_11;
    }
    if ( *((_DWORD *)v14 + 18) == 2 )
      break;
    v14 += 96;
    if ( &v8[96 * v13] == v14 )
    {
      v54 = 0xAAAAAAAAAAAAAAABLL * ((v10 - v14) >> 3);
      goto LABEL_97;
    }
  }
  if ( v10 != v14 + 72 )
    goto LABEL_11;
  while ( 1 )
  {
LABEL_20:
    if ( *(_DWORD *)v8 != 1 )
      goto LABEL_21;
    if ( *((_DWORD *)v8 + 6) != 1 )
    {
      v8 += 24;
      goto LABEL_21;
    }
    if ( *((_DWORD *)v8 + 12) != 1 )
    {
      v8 += 48;
      goto LABEL_21;
    }
    if ( *((_DWORD *)v8 + 18) != 1 )
      break;
    v8 += 96;
    if ( !--v13 )
    {
      v12 = 0xAAAAAAAAAAAAAAABLL * ((v10 - v8) >> 3);
      goto LABEL_101;
    }
  }
  v8 += 72;
LABEL_21:
  if ( v10 == v8 )
  {
LABEL_11:
    if ( *(_BYTE *)(a2 + 61) )
      v15 = sub_3753600(a1, a2, a3);
    else
      v15 = sub_3753880(a1, a2, a3);
    goto LABEL_13;
  }
  if ( *(_BYTE *)(a2 + 60) )
  {
    v75 = 6;
    v61 = (_QWORD *)sub_B0DED0(v61, &v75, 1);
  }
  if ( !*(_BYTE *)(a2 + 61) )
    v61 = (_QWORD *)sub_B0D320(v61);
  v75 = (unsigned __int64)v77;
  v76 = 0x100000000LL;
  v17 = *(_QWORD *)a2;
  if ( !(unsigned int)*(_QWORD *)a2 )
  {
    v58 = 0;
    v35 = 0;
    goto LABEL_44;
  }
  v18 = 0;
  v58 = (unsigned int)v17;
  v66 = 24LL * (unsigned int)v17;
  while ( 2 )
  {
    v19 = (const __m128i *)(*(_QWORD *)(a2 + 8) + v18);
    v68 = _mm_loadu_si128(v19);
    v69 = v19[1].m128i_i64[0];
    if ( v68.m128i_i32[0] == 3 )
    {
      v36 = v68.m128i_i32[2];
      v62 = a1[1];
      if ( !sub_2DADE10(v62, v68.m128i_i32[2]) )
      {
        v40 = v75;
        v71 = v36;
        v41 = (unsigned int)v76;
        v74 = 0;
        v42 = (const __m128i *)&v70;
        v72 = 0;
        v43 = (unsigned int)v76 + 1LL;
        v73 = 0;
        v70 = 0x800000000LL;
        if ( v43 <= HIDWORD(v76) )
          goto LABEL_55;
        if ( v75 > (unsigned __int64)&v70 || (unsigned __int64)&v70 >= v75 + 40LL * (unsigned int)v76 )
        {
          sub_C8D5F0((__int64)&v75, v77, (unsigned int)v76 + 1LL, 0x28u, v43, v37);
          v40 = v75;
          v41 = (unsigned int)v76;
          goto LABEL_55;
        }
        v53 = (signed __int64)&v70 - v75;
LABEL_95:
        sub_C8D5F0((__int64)&v75, v77, v43, 0x28u, v43, v37);
        v40 = v75;
        v41 = (unsigned int)v76;
        v42 = (const __m128i *)(v75 + v53);
        goto LABEL_55;
      }
      if ( v36 < 0 )
        v38 = *(_QWORD *)(*(_QWORD *)(v62 + 56) + 16LL * (v36 & 0x7FFFFFFF) + 8);
      else
        v38 = *(_QWORD *)(*(_QWORD *)(v62 + 304) + 8LL * (unsigned int)v36);
      if ( v38 )
      {
        if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
        {
          v38 = *(_QWORD *)(v38 + 32);
          if ( v38 )
          {
            if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
              BUG();
          }
        }
      }
LABEL_53:
      v39 = *(_QWORD *)(v38 + 16);
      if ( ((*(_WORD *)(v39 + 68) - 12) & 0xFFF7) == 0
        || (v45 = a1[2], v46 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v45 + 520LL), v46 != sub_2DCA430)
        && (v65 = v39, ((void (__fastcall *)(__int64 *, __int64, __int64))v46)(&v70, v45, v39), v39 = v65, (_BYTE)v72) )
      {
        v40 = v75;
        v71 = v36;
        v41 = (unsigned int)v76;
        v74 = 0;
        v42 = (const __m128i *)&v70;
        v72 = 0;
        v43 = (unsigned int)v76 + 1LL;
        v73 = 0;
        v70 = 0x800000000LL;
        if ( v43 <= HIDWORD(v76) )
          goto LABEL_55;
        if ( v75 <= (unsigned __int64)&v70 && (unsigned __int64)&v70 < v75 + 40LL * (unsigned int)v76 )
        {
          v56 = (signed __int64)&v70 - v75;
          sub_C8D5F0((__int64)&v75, v77, (unsigned int)v76 + 1LL, 0x28u, v43, v37);
          v40 = v75;
          v41 = (unsigned int)v76;
          v42 = (const __m128i *)(v75 + v56);
          goto LABEL_55;
        }
        goto LABEL_120;
      }
      v47 = *(_QWORD *)(v39 + 32);
      v48 = v47 + 40LL * (*(_DWORD *)(v39 + 40) & 0xFFFFFF);
      v49 = 0;
      if ( v47 == v48 )
      {
        v49 = 0;
      }
      else
      {
        do
        {
          if ( !*(_BYTE *)v47 && (*(_BYTE *)(v47 + 3) & 0x10) != 0 && v36 == *(_DWORD *)(v47 + 8) )
            break;
          v47 += 40;
          ++v49;
        }
        while ( v48 != v47 );
      }
      v63 = v49;
      v50 = sub_2E8E690(v39);
      v70 = 20;
      v73 = __PAIR64__(v63, v50);
      v30 = (unsigned int)v76;
      v72 = 0;
      v31 = (unsigned int)v76 + 1LL;
      if ( v31 <= HIDWORD(v76) )
      {
        v32 = v75;
        v33 = (const __m128i *)&v70;
        goto LABEL_41;
      }
      v57 = v75;
      if ( v75 <= (unsigned __int64)&v70 && (unsigned __int64)&v70 < v75 + 40LL * (unsigned int)v76 )
      {
LABEL_127:
        sub_C8D5F0((__int64)&v75, v77, v31, 0x28u, v28, v29);
        v32 = v75;
        v30 = (unsigned int)v76;
        v33 = (const __m128i *)((char *)&v70 + v75 - v57);
        goto LABEL_41;
      }
LABEL_143:
      sub_C8D5F0((__int64)&v75, v77, v31, 0x28u, v28, v29);
      v32 = v75;
      v30 = (unsigned int)v76;
      v33 = (const __m128i *)&v70;
      goto LABEL_41;
    }
    if ( v68.m128i_i32[0] )
    {
      sub_3753270(&v70, (__int64)&v68);
      v30 = (unsigned int)v76;
      v31 = (unsigned int)v76 + 1LL;
      if ( v31 <= HIDWORD(v76) )
      {
        v32 = v75;
        v33 = (const __m128i *)&v70;
LABEL_41:
        v34 = (__m128i *)(v32 + 40 * v30);
        *v34 = _mm_loadu_si128(v33);
        v34[1] = _mm_loadu_si128(v33 + 1);
        v34[2].m128i_i64[0] = v33[2].m128i_i64[0];
        LODWORD(v76) = v76 + 1;
        goto LABEL_42;
      }
      v57 = v75;
      if ( v75 <= (unsigned __int64)&v70 && (unsigned __int64)&v70 < v75 + 40LL * (unsigned int)v76 )
        goto LABEL_127;
      goto LABEL_143;
    }
    v20 = *(_BYTE *)(a3 + 8) & 1;
    if ( v20 )
    {
      v21 = a3 + 16;
      v22 = 15;
LABEL_32:
      v23 = 1;
      for ( i = v22 & (v69 + (((unsigned __int64)v68.m128i_i64[1] >> 9) ^ ((unsigned __int64)v68.m128i_i64[1] >> 4)));
            ;
            i = v22 & v26 )
      {
        v25 = v21 + 24LL * i;
        if ( v68.m128i_i64[1] == *(_QWORD *)v25 && (_DWORD)v69 == *(_DWORD *)(v25 + 8) )
          break;
        if ( !*(_QWORD *)v25 && *(_DWORD *)(v25 + 8) == -1 )
        {
          if ( !v20 )
          {
            v27 = *(unsigned int *)(a3 + 24);
            goto LABEL_107;
          }
          v55 = 384;
          goto LABEL_108;
        }
        v26 = v23 + i;
        ++v23;
      }
    }
    else
    {
      v27 = *(unsigned int *)(a3 + 24);
      v21 = *(_QWORD *)(a3 + 16);
      if ( (_DWORD)v27 )
      {
        v22 = v27 - 1;
        goto LABEL_32;
      }
LABEL_107:
      v55 = 24 * v27;
LABEL_108:
      v25 = v21 + v55;
    }
    v51 = 384;
    if ( !v20 )
      v51 = 24LL * *(unsigned int *)(a3 + 24);
    if ( v25 != v21 + v51 )
    {
      v36 = sub_3752000(a1, v68.m128i_u64[1], v69, a3, v21, (unsigned int)v69);
      v64 = a1[1];
      if ( sub_2DADE10(v64, v36) )
      {
        if ( v36 < 0 )
          v38 = *(_QWORD *)(*(_QWORD *)(v64 + 56) + 16LL * (v36 & 0x7FFFFFFF) + 8);
        else
          v38 = *(_QWORD *)(*(_QWORD *)(v64 + 304) + 8LL * (unsigned int)v36);
        if ( v38 )
        {
          if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
          {
            v38 = *(_QWORD *)(v38 + 32);
            if ( v38 )
            {
              if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
                BUG();
            }
          }
        }
        goto LABEL_53;
      }
      v40 = v75;
      v71 = v36;
      v41 = (unsigned int)v76;
      v74 = 0;
      v42 = (const __m128i *)&v70;
      v72 = 0;
      v43 = (unsigned int)v76 + 1LL;
      v73 = 0;
      v70 = 0x800000000LL;
      if ( v43 <= HIDWORD(v76) )
      {
LABEL_55:
        v44 = (__m128i *)(v40 + 40 * v41);
        *v44 = _mm_loadu_si128(v42);
        v44[1] = _mm_loadu_si128(v42 + 1);
        v44[2].m128i_i64[0] = v42[2].m128i_i64[0];
        LODWORD(v76) = v76 + 1;
LABEL_42:
        v18 += 24;
        if ( v66 == v18 )
          break;
        continue;
      }
      if ( v75 <= (unsigned __int64)&v70 && (unsigned __int64)&v70 < v75 + 40LL * (unsigned int)v76 )
      {
        v53 = (signed __int64)&v70 - v75;
        goto LABEL_95;
      }
LABEL_120:
      sub_C8D5F0((__int64)&v75, v77, (unsigned int)v76 + 1LL, 0x28u, v43, v37);
      v40 = v75;
      v41 = (unsigned int)v76;
      v42 = (const __m128i *)&v70;
      goto LABEL_55;
    }
    break;
  }
  v35 = (unsigned int)v76;
LABEL_44:
  if ( v58 == v35 )
  {
    sub_2E908B0((_QWORD *)*a1, &v67, (_WORD *)(v59 - 640), 0, (const __m128i *)v75, v58, v60, (__int64)v61);
    v15 = v52;
  }
  else
  {
    v15 = sub_3753560((__int64)a1, a2);
  }
  if ( (_BYTE *)v75 != v77 )
    _libc_free(v75);
LABEL_13:
  if ( v67 )
    sub_B91220((__int64)&v67, (__int64)v67);
  return v15;
}
