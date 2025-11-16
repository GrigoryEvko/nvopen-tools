// Function: sub_F04320
// Address: 0xf04320
//
__int64 *__fastcall sub_F04320(__int64 *a1, unsigned __int64 a2, int a3, char a4, unsigned int a5)
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
  _DWORD *v39; // r13
  _DWORD *v40; // rax
  _QWORD *v41; // rbx
  __m128i *v42; // rsi
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
  __int64 v59; // rax
  _QWORD *i; // rbx
  unsigned __int64 v61; // rcx
  char v62; // dl
  int v63; // esi
  unsigned __int64 v64; // [rsp+0h] [rbp-100h]
  unsigned __int64 v65; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v66; // [rsp+10h] [rbp-F0h]
  char v67; // [rsp+18h] [rbp-E8h]
  int v68; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v69; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v70; // [rsp+20h] [rbp-E0h]
  int v71; // [rsp+20h] [rbp-E0h]
  char *v74; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v75; // [rsp+48h] [rbp-B8h]
  _QWORD v76[2]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD *v77; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v78; // [rsp+68h] [rbp-98h]
  _QWORD v79[2]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v80; // [rsp+80h] [rbp-80h] BYREF
  _QWORD *v81; // [rsp+88h] [rbp-78h]
  _QWORD v82[2]; // [rsp+90h] [rbp-70h] BYREF
  __m128i *v83; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v84; // [rsp+A8h] [rbp-58h]
  __m128i v85; // [rsp+B0h] [rbp-50h] BYREF

  if ( !a2 )
  {
    *a1 = (__int64)(a1 + 2);
    sub_F04030(a1, "0.0", (__int64)"");
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
        v74 = (char *)v76;
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
        v74 = (char *)v76;
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
        LOBYTE(v76[0]) = 0;
        v74 = (char *)v76;
        v75 = 0;
        if ( v23 )
        {
          v5 = a2 >> -(char)a3;
LABEL_4:
          v67 = a4;
          v8 = (char *)v76;
          v70 = v7;
          v9 = 0;
          while ( 1 )
          {
            v10 = v9 + 1;
            v11 = 15;
            if ( v8 != (char *)v76 )
              v11 = v76[0];
            if ( v10 > v11 )
            {
              sub_2240BB0(&v74, v9, 0, 0, 1);
              v8 = v74;
            }
            v8[v9] = v5 % 0xA + 48;
            v75 = v9 + 1;
            v74[v10] = 0;
            if ( v5 <= 9 )
              break;
            v9 = v75;
            v8 = v74;
            v5 /= 0xAu;
          }
          v16 = v75;
          v17 = v74;
          v7 = v70;
          a4 = v67;
          if ( &v74[v75] == v74 )
          {
            v21 = v75;
            v22 = 0;
            v23 = 0;
          }
          else
          {
            v18 = &v74[v75 - 1];
            if ( v18 <= v74 )
            {
              v21 = v75;
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
              v21 = v75;
              v17 = v74;
              v22 = 0;
              v23 = 0;
            }
          }
LABEL_24:
          if ( !v7 )
          {
            *a1 = (__int64)(a1 + 2);
            sub_F03F80(a1, v17, (__int64)&v17[v21]);
            if ( a1[1] == 0x3FFFFFFFFFFFFFFFLL || a1[1] == 4611686018427387902LL )
              sub_4262D8((__int64)"basic_string::append");
            sub_2241490(a1, ".0", 2, v24);
LABEL_27:
            if ( v74 != (char *)v76 )
              j_j___libc_free_0(v74, v76[0] + 1LL);
            return a1;
          }
          v25 = 15;
          v26 = v21 + 1;
          if ( v17 != (char *)v76 )
            v25 = v76[0];
          if ( v26 > v25 )
          {
            v69 = v16;
            v71 = v22;
            sub_2240BB0(&v74, v21, 0, 0, 1);
            v17 = v74;
            v16 = v69;
            v22 = v71;
          }
          v17[v21] = 46;
          v27 = 0;
          v28 = 64 - a4;
          v75 = v21 + 1;
          v29 = v7 >> 4;
          v74[v26] = 0;
          v30 = v75;
          v31 = 1LL << v28;
          v64 = v75;
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
            v35 = v74;
            v36 = 15;
            if ( v74 != (char *)v76 )
              v36 = v76[0];
            if ( v33 > v36 )
            {
              v65 = v27;
              v66 = v16;
              v68 = v22;
              sub_2240BB0(&v74, v30, 0, 0, 1);
              v35 = v74;
              v27 = v65;
              v16 = v66;
              v22 = v68;
            }
            v35[v30] = (v34 >> 60) + 48;
            v29 = v34 & 0xFFFFFFFFFFFFFFFLL;
            v75 = v30 + 1;
            v74[v33] = 0;
            if ( v16 || v74[v75 - 1] != 48 )
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
            v30 = v75;
          }
          if ( !a5 || (v37 = a5, a5 >= v16) )
          {
LABEL_52:
            sub_F040E0(a1, &v74);
            goto LABEL_27;
          }
LABEL_78:
          v44 = v75;
          v45 = v75 + v37 - v16;
          v46 = v64 + 1;
          if ( v45 >= v64 + 1 )
            v46 = v45;
          if ( v75 <= v46 )
            goto LABEL_52;
          v47 = v74;
          v48 = &v74[v46];
          if ( (unsigned int)(v74[v46] - 53) > 4 )
          {
            v83 = &v85;
            sub_F04030((__int64 *)&v83, v74, (__int64)&v74[v46]);
            sub_F040E0(a1, &v83);
            if ( v83 != &v85 )
              j_j___libc_free_0(v83, v85.m128i_i64[0] + 1);
            goto LABEL_27;
          }
          if ( v74 == v48 )
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
                  v44 = v75;
                  v48 = v74;
                  goto LABEL_86;
                }
                *(v48 - 1) = 48;
              }
              --v48;
            }
            while ( v47 != v48 );
            v44 = v75;
            v48 = v74;
            v50 = 1;
          }
LABEL_86:
          if ( v46 > v44 )
            v46 = v44;
          v80 = v82;
          sub_F04030((__int64 *)&v80, v48, (__int64)&v48[v46]);
          v77 = v79;
          sub_2240A50(&v77, v50, 49, v51, v52);
          v53 = 15;
          v54 = 15;
          if ( v77 != v79 )
            v54 = v79[0];
          v55 = (unsigned __int64)v81 + v78;
          if ( (unsigned __int64)v81 + v78 <= v54 )
            goto LABEL_94;
          if ( v80 != v82 )
            v53 = v82[0];
          if ( v55 <= v53 )
          {
            v56 = (__m128i *)sub_2241130(&v80, 0, 0, v77, v78);
            v83 = &v85;
            v57 = v56->m128i_i64[0];
            v58 = v56 + 1;
            if ( (__m128i *)v56->m128i_i64[0] != &v56[1] )
              goto LABEL_95;
          }
          else
          {
LABEL_94:
            v56 = (__m128i *)sub_2241490(&v77, v80, v81, v55);
            v83 = &v85;
            v57 = v56->m128i_i64[0];
            v58 = v56 + 1;
            if ( (__m128i *)v56->m128i_i64[0] != &v56[1] )
            {
LABEL_95:
              v83 = (__m128i *)v57;
              v85.m128i_i64[0] = v56[1].m128i_i64[0];
LABEL_96:
              v84 = v56->m128i_i64[1];
              v56->m128i_i64[0] = (__int64)v58;
              v56->m128i_i64[1] = 0;
              v56[1].m128i_i8[0] = 0;
              sub_F040E0(a1, &v83);
              if ( v83 != &v85 )
                j_j___libc_free_0(v83, v85.m128i_i64[0] + 1);
              if ( v77 != v79 )
                j_j___libc_free_0(v77, v79[0] + 1LL);
              if ( v80 != v82 )
                j_j___libc_free_0(v80, v82[0] + 1LL);
              goto LABEL_27;
            }
          }
          v85 = _mm_loadu_si128(v56 + 1);
          goto LABEL_96;
        }
        v22 = 0;
LABEL_57:
        v17 = (char *)v76;
        v75 = 1;
        v21 = 1;
        LOWORD(v76[0]) = 48;
        v16 = 0;
        goto LABEL_24;
      }
    }
    _BitScanReverse64(&v61, a2);
    v38 = a3 + 63 - (v61 ^ 0x3F);
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
    LODWORD(a2) = v63 ^ 0x3F;
LABEL_63:
    v38 = a3 + 63 - a2;
    if ( v38 > 16382 )
    {
      v14 = 32766;
      v5 <<= (unsigned __int8)a3 + 64;
      goto LABEL_65;
    }
LABEL_109:
    v62 = a3 - v38;
    v14 = (unsigned int)(v38 + 0x3FFF);
    v5 <<= v62 + 63;
LABEL_65:
    if ( (v5 & 0x8000000000000000LL) == 0LL )
      v14 = 0;
    goto LABEL_67;
  }
  if ( v5 )
  {
LABEL_3:
    LOBYTE(v76[0]) = 0;
    v7 = 0;
    v74 = (char *)v76;
    v75 = 0;
    goto LABEL_4;
  }
  v14 = 0;
LABEL_67:
  v77 = (_QWORD *)v5;
  v78 = v14;
  sub_C438C0((__int64)&v83, 80, &v77, 2u);
  v39 = sub_C33420();
  v40 = sub_C33340();
  v41 = v40;
  if ( v39 == v40 )
    sub_C3C640(&v80, (__int64)v40, &v83);
  else
    sub_C3B160((__int64)&v80, v39, (__int64 *)&v83);
  if ( (unsigned int)v84 > 0x40 && v83 )
    j_j___libc_free_0_0(v83);
  v83 = (__m128i *)&v85.m128i_u64[1];
  v84 = 0;
  v85.m128i_i64[0] = 24;
  if ( v80 == v41 )
    sub_C40650((__int64)&v80, (__int64 *)&v83, a5, 0, 1u);
  else
    sub_C35AD0((__int64)&v80, (__int64 *)&v83, a5, 0, 1);
  v42 = v83;
  v43 = v84;
  *a1 = (__int64)(a1 + 2);
  sub_F03F80(a1, v42, (__int64)v42->m128i_i64 + v43);
  if ( v83 != (__m128i *)&v85.m128i_u64[1] )
    _libc_free(v83, v42);
  if ( v41 == v80 )
  {
    if ( v81 )
    {
      v59 = 3LL * *(v81 - 1);
      for ( i = &v81[v59]; v81 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v80);
  }
  return a1;
}
