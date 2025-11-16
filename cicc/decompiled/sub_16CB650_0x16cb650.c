// Function: sub_16CB650
// Address: 0x16cb650
//
__int64 *__fastcall sub_16CB650(__int64 *a1, unsigned __int64 a2, int a3, char a4, unsigned int a5)
{
  unsigned __int64 v5; // r12
  unsigned __int64 v7; // r14
  char *v8; // rcx
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  int v13; // ecx
  __int64 v14; // rax
  unsigned __int64 v16; // rax
  char *v17; // rsi
  char *v18; // rdx
  char v19; // cl
  char v20; // di
  __int64 v21; // r15
  int v22; // r10d
  unsigned __int64 v23; // r13
  __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // r9
  char v28; // cl
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // r12
  __int64 v32; // r13
  unsigned __int64 v33; // r14
  unsigned __int64 v34; // rbx
  char *v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  int v38; // eax
  void *v39; // r12
  void *v40; // rax
  void *v41; // rbx
  _BYTE *v42; // rsi
  __int64 v43; // rdx
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rax
  char *v47; // r8
  char *v48; // rsi
  char v49; // dl
  __int64 v50; // r14
  __int64 v51; // rcx
  __int64 v52; // r8
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rcx
  __m128i *v56; // rax
  __int64 v57; // rcx
  __m128i *v58; // rdx
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // rbx
  unsigned __int64 v62; // rcx
  char v63; // dl
  int v64; // esi
  unsigned __int64 v65; // [rsp+0h] [rbp-100h]
  unsigned __int64 v66; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v67; // [rsp+10h] [rbp-F0h]
  char v68; // [rsp+18h] [rbp-E8h]
  int v69; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v70; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v71; // [rsp+20h] [rbp-E0h]
  int v72; // [rsp+20h] [rbp-E0h]
  char *v75; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v76; // [rsp+48h] [rbp-B8h]
  _QWORD v77[2]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD *v78; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+68h] [rbp-98h]
  _QWORD v80[2]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v81; // [rsp+80h] [rbp-80h] BYREF
  void *v82; // [rsp+88h] [rbp-78h] BYREF
  _QWORD v83[2]; // [rsp+90h] [rbp-70h] BYREF
  _OWORD *v84; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-58h]
  _OWORD v86[5]; // [rsp+B0h] [rbp-50h] BYREF

  if ( !a2 )
  {
    *a1 = (__int64)(a1 + 2);
    sub_16CB2B0(a1, "0.0", (__int64)"");
    return a1;
  }
  v5 = a2;
  if ( !(_WORD)a3 )
    goto LABEL_3;
  if ( (__int16)a3 <= 0 )
  {
    if ( (__int16)a3 < -63 )
    {
      if ( (_WORD)a3 == 0xFFC0 )
      {
        v7 = a2;
        v22 = 0;
        v23 = 0;
        v75 = (char *)v77;
        goto LABEL_57;
      }
      if ( (__int16)a3 < -119 )
      {
        _BitScanReverse64(&a2, a2);
        LODWORD(a2) = a2 ^ 0x3F;
LABEL_62:
        a3 = (__int16)a3;
        goto LABEL_63;
      }
      a3 = (__int16)a3;
      v22 = -64 - (__int16)a3;
      v7 = a2 >> (-64 - (unsigned __int8)a3);
      v23 = a2 << ((unsigned __int8)a3 + 0x80);
      if ( v7 )
      {
        v75 = (char *)v77;
        goto LABEL_57;
      }
    }
    else
    {
      a3 = (__int16)a3;
      v23 = a2 >> -(char)a3;
      v7 = a2 << ((unsigned __int8)a3 + 64);
      if ( v7 | v23 )
      {
        LOBYTE(v77[0]) = 0;
        v75 = (char *)v77;
        v76 = 0;
        if ( v23 )
        {
          v5 = a2 >> -(char)a3;
LABEL_4:
          v68 = a4;
          v8 = (char *)v77;
          v71 = v7;
          v9 = 0;
          while ( 1 )
          {
            v10 = v9 + 1;
            v11 = 15;
            if ( v8 != (char *)v77 )
              v11 = v77[0];
            if ( v10 > v11 )
            {
              sub_2240BB0(&v75, v9, 0, 0, 1);
              v8 = v75;
            }
            v8[v9] = v5 % 0xA + 48;
            v76 = v9 + 1;
            v75[v10] = 0;
            if ( v5 <= 9 )
              break;
            v9 = v76;
            v8 = v75;
            v5 /= 0xAu;
          }
          v16 = v76;
          v17 = v75;
          v7 = v71;
          a4 = v68;
          if ( v75 == &v75[v76] )
          {
            v21 = v76;
            v22 = 0;
            v23 = 0;
          }
          else
          {
            v18 = &v75[v76 - 1];
            if ( v75 >= v18 )
            {
              v21 = v76;
              v23 = 0;
              v22 = 0;
            }
            else
            {
              do
              {
                v19 = *v17;
                v20 = *v18;
                ++v17;
                --v18;
                *(v17 - 1) = v20;
                v18[1] = v19;
              }
              while ( v18 > v17 );
              v21 = v76;
              v17 = v75;
              v22 = 0;
              v23 = 0;
            }
          }
LABEL_24:
          if ( !v7 )
          {
            *a1 = (__int64)(a1 + 2);
            sub_16CB360(a1, v17, (__int64)&v17[v21]);
            if ( a1[1] == 0x3FFFFFFFFFFFFFFFLL || a1[1] == 4611686018427387902LL )
              sub_4262D8((__int64)"basic_string::append");
            sub_2241490(a1, ".0", 2, v24);
LABEL_27:
            if ( v75 != (char *)v77 )
              j_j___libc_free_0(v75, v77[0] + 1LL);
            return a1;
          }
          v25 = 15;
          v26 = v21 + 1;
          if ( v17 != (char *)v77 )
            v25 = v77[0];
          if ( v26 > v25 )
          {
            v70 = v16;
            v72 = v22;
            sub_2240BB0(&v75, v21, 0, 0, 1);
            v17 = v75;
            v16 = v70;
            v22 = v72;
          }
          v17[v21] = 46;
          v27 = 0;
          v28 = 64 - a4;
          v76 = v21 + 1;
          v29 = v7 >> 4;
          v75[v26] = 0;
          v30 = v76;
          v31 = 1LL << v28;
          v65 = v76;
          v32 = (v7 << 56) & 0xF00000000000000LL | (v23 >> 8);
          while ( 1 )
          {
            if ( v22 )
            {
              --v22;
              v31 *= 5LL;
            }
            else
            {
              v31 *= 10LL;
            }
            v33 = v30 + 1;
            v34 = ((unsigned __int64)(5 * v32) >> 59) + 10 * v29;
            v32 = (10 * v32) & 0xFFFFFFFFFFFFFFFLL;
            v35 = v75;
            v36 = 15;
            if ( v75 != (char *)v77 )
              v36 = v77[0];
            if ( v33 > v36 )
            {
              v66 = v27;
              v67 = v16;
              v69 = v22;
              sub_2240BB0(&v75, v30, 0, 0, 1);
              v35 = v75;
              v27 = v66;
              v16 = v67;
              v22 = v69;
            }
            v35[v30] = (v34 >> 60) + 48;
            v29 = v34 & 0xFFFFFFFFFFFFFFFLL;
            v76 = v30 + 1;
            v75[v33] = 0;
            if ( v16 || v75[v76 - 1] != 48 )
              ++v16;
            ++v27;
            if ( !v31 || 16 * v29 < v31 >> 1 )
              break;
            if ( a5 )
            {
              v37 = a5;
              if ( a5 < v16 && v27 > 1 )
                goto LABEL_78;
            }
            v30 = v76;
          }
          if ( !a5 || (v37 = a5, a5 >= v16) )
          {
LABEL_52:
            sub_16CB410(a1, &v75);
            goto LABEL_27;
          }
LABEL_78:
          v44 = v76;
          v45 = v76 + v37 - v16;
          v46 = v65 + 1;
          if ( v45 >= v65 + 1 )
            v46 = v45;
          if ( v76 <= v46 )
            goto LABEL_52;
          v47 = v75;
          v48 = &v75[v46];
          if ( (unsigned int)(v75[v46] - 53) > 4 )
          {
            v84 = v86;
            sub_16CB2B0((__int64 *)&v84, v75, (__int64)&v75[v46]);
            sub_16CB410(a1, &v84);
            if ( v84 != v86 )
              j_j___libc_free_0(v84, *(_QWORD *)&v86[0] + 1LL);
            goto LABEL_27;
          }
          if ( v75 == v48 )
          {
            v50 = 1;
          }
          else
          {
            do
            {
              v49 = *(v48 - 1);
              if ( v49 != 46 )
              {
                if ( v49 != 57 )
                {
                  v50 = 0;
                  *(v48 - 1) = v49 + 1;
                  v44 = v76;
                  v48 = v75;
                  goto LABEL_86;
                }
                *(v48 - 1) = 48;
              }
              --v48;
            }
            while ( v47 != v48 );
            v44 = v76;
            v48 = v75;
            v50 = 1;
          }
LABEL_86:
          if ( v46 > v44 )
            v46 = v44;
          v81 = v83;
          sub_16CB2B0((__int64 *)&v81, v48, (__int64)&v48[v46]);
          v78 = v80;
          sub_2240A50(&v78, v50, 49, v51, v52);
          v53 = 15;
          v54 = 15;
          if ( v78 != v80 )
            v54 = v80[0];
          v55 = (unsigned __int64)v82 + v79;
          if ( (unsigned __int64)v82 + v79 <= v54 )
            goto LABEL_94;
          if ( v81 != v83 )
            v53 = v83[0];
          if ( v55 <= v53 )
          {
            v56 = (__m128i *)sub_2241130(&v81, 0, 0, v78, v79);
            v84 = v86;
            v57 = v56->m128i_i64[0];
            v58 = v56 + 1;
            if ( (__m128i *)v56->m128i_i64[0] != &v56[1] )
              goto LABEL_95;
          }
          else
          {
LABEL_94:
            v56 = (__m128i *)sub_2241490(&v78, v81, v82, v55);
            v84 = v86;
            v57 = v56->m128i_i64[0];
            v58 = v56 + 1;
            if ( (__m128i *)v56->m128i_i64[0] != &v56[1] )
            {
LABEL_95:
              v84 = (_OWORD *)v57;
              *(_QWORD *)&v86[0] = v56[1].m128i_i64[0];
LABEL_96:
              v85 = v56->m128i_i64[1];
              v56->m128i_i64[0] = (__int64)v58;
              v56->m128i_i64[1] = 0;
              v56[1].m128i_i8[0] = 0;
              sub_16CB410(a1, &v84);
              if ( v84 != v86 )
                j_j___libc_free_0(v84, *(_QWORD *)&v86[0] + 1LL);
              if ( v78 != v80 )
                j_j___libc_free_0(v78, v80[0] + 1LL);
              if ( v81 != v83 )
                j_j___libc_free_0(v81, v83[0] + 1LL);
              goto LABEL_27;
            }
          }
          v86[0] = _mm_loadu_si128(v56 + 1);
          goto LABEL_96;
        }
        v22 = 0;
LABEL_57:
        v17 = (char *)v77;
        v76 = 1;
        v21 = 1;
        LOWORD(v77[0]) = 48;
        v16 = 0;
        goto LABEL_24;
      }
    }
    _BitScanReverse64(&v62, a2);
    v38 = a3 + 63 - (v62 ^ 0x3F);
    goto LABEL_109;
  }
  _BitScanReverse64(&v12, a2);
  v13 = v12 ^ 0x3F;
  LODWORD(a2) = v13;
  if ( (__int16)a3 < (__int16)v13 )
    LOWORD(v13) = a3;
  if ( !(_WORD)v13 )
    goto LABEL_62;
  v5 <<= v13;
  LOWORD(a3) = a3 - v13;
  if ( (_WORD)a3 )
  {
    a3 = (__int16)a3;
    v14 = 0;
    if ( !v5 )
      goto LABEL_67;
    _BitScanReverse64(&a2, v5);
    LODWORD(a2) = v64 ^ 0x3F;
LABEL_63:
    v38 = a3 + 63 - a2;
    if ( v38 > 16382 )
    {
      v14 = 32766;
      v5 <<= (unsigned __int8)a3 + 64;
      goto LABEL_65;
    }
LABEL_109:
    v63 = a3 - v38;
    v14 = (unsigned int)(v38 + 0x3FFF);
    v5 <<= v63 + 63;
LABEL_65:
    if ( (v5 & 0x8000000000000000LL) == 0LL )
      v14 = 0;
    goto LABEL_67;
  }
  if ( v5 )
  {
LABEL_3:
    LOBYTE(v77[0]) = 0;
    v7 = 0;
    v75 = (char *)v77;
    v76 = 0;
    goto LABEL_4;
  }
  v14 = 0;
LABEL_67:
  v78 = (_QWORD *)v5;
  v79 = v14;
  sub_16A50F0((__int64)&v84, 80, &v78, 2u);
  v39 = sub_16982A0();
  v40 = sub_16982C0();
  v41 = v40;
  if ( v39 == v40 )
    sub_169D060(&v82, (__int64)v40, (__int64 *)&v84);
  else
    sub_169D050((__int64)&v82, v39, (__int64 *)&v84);
  if ( (unsigned int)v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  v84 = v86;
  v85 = 0x1800000000LL;
  if ( v82 == v41 )
    sub_16A4A90((__int64)&v82, (__int64)&v84, a5, 0, 1u);
  else
    sub_16A3760((__int64)&v82, (__int64)&v84, a5, 0, 1);
  v42 = v84;
  v43 = (unsigned int)v85;
  *a1 = (__int64)(a1 + 2);
  sub_16CB360(a1, v42, (__int64)&v42[v43]);
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
  if ( v41 == v82 )
  {
    v59 = v83[0];
    if ( v83[0] )
    {
      v60 = 32LL * *(_QWORD *)(v83[0] - 8LL);
      v61 = v83[0] + v60;
      if ( v83[0] != v83[0] + v60 )
      {
        do
        {
          v61 -= 32;
          sub_127D120((_QWORD *)(v61 + 8));
        }
        while ( v59 != v61 );
      }
      j_j_j___libc_free_0_0(v59 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v82);
  }
  return a1;
}
