// Function: sub_16F6C30
// Address: 0x16f6c30
//
_QWORD *__fastcall sub_16F6C30(_QWORD *a1, char *a2, __int64 a3, unsigned __int64 a4)
{
  char *v4; // rbx
  char *v5; // r15
  char v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  unsigned __int64 v11; // r14
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __m128i *v14; // rsi
  __int64 v15; // r8
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  char *v21; // rsi
  char v22; // dl
  __int64 v23; // rcx
  __int64 v24; // r8
  __m128i *v25; // rax
  __int64 v26; // rcx
  __m128i *v27; // rax
  char v28; // dl
  __m128i *v29; // rax
  __int64 v30; // rcx
  __m128i *v31; // rax
  unsigned __int64 v32; // rsi
  __m128i *v33; // rdx
  __m128i *v34; // rax
  __int64 v35; // rcx
  __m128i *v36; // rax
  __int64 v37; // rcx
  char v38; // [rsp+27h] [rbp-B9h]
  _QWORD *v39; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v40; // [rsp+38h] [rbp-A8h]
  _QWORD v41[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD v42[2]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v43[2]; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v44; // [rsp+70h] [rbp-70h] BYREF
  __int64 v45; // [rsp+78h] [rbp-68h]
  __m128i v46; // [rsp+80h] [rbp-60h] BYREF
  __m128i *v47; // [rsp+90h] [rbp-50h] BYREF
  __int64 v48; // [rsp+98h] [rbp-48h]
  __m128i v49; // [rsp+A0h] [rbp-40h] BYREF

  v4 = &a2[a3];
  v38 = a4;
  *a1 = a1 + 2;
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  if ( &a2[a3] == a2 )
    return a1;
  v5 = a2;
  while ( 1 )
  {
    v6 = *v5;
    if ( *v5 == 92 )
    {
      while ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
      {
        sub_2241490(a1, "\\\\", 2, a4);
LABEL_34:
        if ( v4 == ++v5 )
          return a1;
        v6 = *v5;
        if ( *v5 != 92 )
          goto LABEL_4;
      }
LABEL_118:
      sub_4262D8((__int64)"basic_string::append");
    }
LABEL_4:
    switch ( v6 )
    {
      case 34:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\\"", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 0:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\0", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 7:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\a", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 8:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\b", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 9:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\t", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 10:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\n", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 11:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\v", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 12:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\f", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 13:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\r", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
      case 27:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) > 1 )
        {
          sub_2241490(a1, "\\e", 2, a4);
          goto LABEL_34;
        }
        goto LABEL_118;
    }
    if ( (unsigned __int8)v6 <= 0x1Fu )
    {
      v20 = v6;
      v21 = &v49.m128i_i8[1];
      do
      {
        --v21;
        v22 = (v20 & 0xF) + 55;
        if ( (v20 & 0xF) <= 9 )
          v22 = (v20 & 0xF) + 48;
        v20 >>= 4;
        *v21 = v22;
      }
      while ( v20 );
      v39 = v41;
      sub_16F6740((__int64 *)&v39, v21, (__int64)v49.m128i_i64 + 1);
      v42[0] = v43;
      sub_2240A50(v42, 2 - v40, 48, v23, v24);
      v25 = (__m128i *)sub_2241130(v42, 0, 0, "\\x", 2);
      v44 = &v46;
      if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
      {
        v46 = _mm_loadu_si128(v25 + 1);
      }
      else
      {
        v44 = (__m128i *)v25->m128i_i64[0];
        v46.m128i_i64[0] = v25[1].m128i_i64[0];
      }
      v45 = v25->m128i_i64[1];
      v26 = v45;
      v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
      v25->m128i_i64[1] = 0;
      v25[1].m128i_i8[0] = 0;
      v27 = (__m128i *)sub_2241490(&v44, v39, v40, v26);
      v47 = &v49;
      if ( (__m128i *)v27->m128i_i64[0] == &v27[1] )
      {
        v49 = _mm_loadu_si128(v27 + 1);
      }
      else
      {
        v47 = (__m128i *)v27->m128i_i64[0];
        v49.m128i_i64[0] = v27[1].m128i_i64[0];
      }
      v48 = v27->m128i_i64[1];
      v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
      v27->m128i_i64[1] = 0;
      v27[1].m128i_i8[0] = 0;
      sub_2241490(a1, v47, v48, &v49);
      a4 = (unsigned __int64)&v49;
      if ( v47 != &v49 )
        j_j___libc_free_0(v47, v49.m128i_i64[0] + 1);
      if ( v44 != &v46 )
        j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
      if ( (_QWORD *)v42[0] != v43 )
        j_j___libc_free_0(v42[0], v43[0] + 1LL);
      if ( v39 != v41 )
        j_j___libc_free_0(v39, v41[0] + 1LL);
      goto LABEL_34;
    }
    if ( v6 >= 0 )
    {
      v17 = a1[1];
      v18 = *a1;
      v19 = v17 + 1;
      if ( a1 + 2 == (_QWORD *)*a1 )
        a4 = 15;
      else
        a4 = a1[2];
      if ( v19 > a4 )
      {
        sub_2240BB0(a1, v17, 0, 0, 1);
        v18 = *a1;
        v19 = v17 + 1;
      }
      *(_BYTE *)(v18 + v17) = v6;
      a1[1] = v19;
      *(_BYTE *)(*a1 + v17 + 1) = 0;
      goto LABEL_34;
    }
    v11 = sub_16F61C0(v5, v4 - v5);
    if ( !HIDWORD(v11) )
      break;
    switch ( (_DWORD)v11 )
    {
      case 0x85:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_118;
        sub_2241490(a1, "\\N", 2, v8);
        break;
      case 0xA0:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_118;
        sub_2241490(a1, "\\_", 2, v8);
        break;
      case 0x2028:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_118;
        sub_2241490(a1, "\\L", 2, v8);
        break;
      case 0x2029:
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) <= 1 )
          goto LABEL_118;
        sub_2241490(a1, "\\P", 2, v8);
        break;
      default:
        if ( v38 || !(unsigned __int8)sub_16FF6D0((unsigned int)v11) )
        {
          v13 = (unsigned int)v11;
          if ( (_DWORD)v11 )
          {
            v14 = (__m128i *)&v49.m128i_i8[1];
            do
            {
              v14 = (__m128i *)((char *)v14 - 1);
              v28 = (v13 & 0xF) + 55;
              if ( (v13 & 0xF) <= 9 )
                v28 = (v13 & 0xF) + 48;
              v13 >>= 4;
              v14->m128i_i8[0] = v28;
            }
            while ( v13 );
          }
          else
          {
            v49.m128i_i8[0] = 48;
            v14 = &v49;
          }
          v39 = v41;
          sub_16F6740((__int64 *)&v39, v14, (__int64)v49.m128i_i64 + 1);
          if ( v40 <= 2 )
          {
            v42[0] = v43;
            sub_2240A50(v42, 2 - v40, 48, v43, v15);
            v34 = (__m128i *)sub_2241130(v42, 0, 0, "\\x", 2);
            v44 = &v46;
            if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
            {
              v46 = _mm_loadu_si128(v34 + 1);
            }
            else
            {
              v44 = (__m128i *)v34->m128i_i64[0];
              v46.m128i_i64[0] = v34[1].m128i_i64[0];
            }
            v45 = v34->m128i_i64[1];
            v35 = v45;
            v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
            v34->m128i_i64[1] = 0;
            v34[1].m128i_i8[0] = 0;
            v31 = (__m128i *)sub_2241490(&v44, v39, v40, v35);
            v47 = &v49;
            v32 = v31->m128i_i64[0];
            v33 = v31 + 1;
            if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
              goto LABEL_106;
          }
          else
          {
            if ( v40 > 4 )
            {
              if ( v40 > 8 )
              {
LABEL_28:
                if ( v39 != v41 )
                  j_j___libc_free_0(v39, v41[0] + 1LL);
                break;
              }
              v42[0] = v43;
              sub_2240A50(v42, 8 - v40, 48, a4, v15);
              v29 = (__m128i *)sub_2241130(v42, 0, 0, "\\U", 2);
              v44 = &v46;
              if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
              {
                v46 = _mm_loadu_si128(v29 + 1);
              }
              else
              {
                v44 = (__m128i *)v29->m128i_i64[0];
                v46.m128i_i64[0] = v29[1].m128i_i64[0];
              }
              v45 = v29->m128i_i64[1];
              v30 = v45;
              v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
              v29->m128i_i64[1] = 0;
              v29[1].m128i_i8[0] = 0;
              v31 = (__m128i *)sub_2241490(&v44, v39, v40, v30);
              v47 = &v49;
              v32 = v31->m128i_i64[0];
              v33 = v31 + 1;
              if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
                goto LABEL_96;
LABEL_106:
              v49 = _mm_loadu_si128(v31 + 1);
              goto LABEL_97;
            }
            v42[0] = v43;
            sub_2240A50(v42, 4 - v40, 48, v43, v15);
            v36 = (__m128i *)sub_2241130(v42, 0, 0, "\\u", 2);
            v44 = &v46;
            if ( (__m128i *)v36->m128i_i64[0] == &v36[1] )
            {
              v46 = _mm_loadu_si128(v36 + 1);
            }
            else
            {
              v44 = (__m128i *)v36->m128i_i64[0];
              v46.m128i_i64[0] = v36[1].m128i_i64[0];
            }
            v45 = v36->m128i_i64[1];
            v37 = v45;
            v36->m128i_i64[0] = (__int64)v36[1].m128i_i64;
            v36->m128i_i64[1] = 0;
            v36[1].m128i_i8[0] = 0;
            v31 = (__m128i *)sub_2241490(&v44, v39, v40, v37);
            v47 = &v49;
            v32 = v31->m128i_i64[0];
            v33 = v31 + 1;
            if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
              goto LABEL_106;
          }
LABEL_96:
          v47 = (__m128i *)v32;
          v49.m128i_i64[0] = v31[1].m128i_i64[0];
LABEL_97:
          v48 = v31->m128i_i64[1];
          v31->m128i_i64[0] = (__int64)v33;
          v31->m128i_i64[1] = 0;
          v31[1].m128i_i8[0] = 0;
          sub_2241490(a1, v47, v48, &v49);
          a4 = (unsigned __int64)&v49;
          if ( v47 != &v49 )
            j_j___libc_free_0(v47, v49.m128i_i64[0] + 1);
          if ( v44 != &v46 )
            j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
          if ( (_QWORD *)v42[0] != v43 )
            j_j___libc_free_0(v42[0], v43[0] + 1LL);
          goto LABEL_28;
        }
        if ( HIDWORD(v11) > (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a1[1]) )
          goto LABEL_118;
        sub_2241490(a1, v5, HIDWORD(v11), v12);
        break;
    }
    v5 += (unsigned int)(HIDWORD(v11) - 1) + 1;
    if ( v4 == v5 )
      return a1;
  }
  v48 = 0x400000000LL;
  v47 = &v49;
  sub_16F69C0(0xFFFDu, (__int64)&v47, v7, v8, v9, v10);
  sub_2241130(a1, a1[1], 0, v47, (unsigned int)v48);
  if ( v47 != &v49 )
    _libc_free((unsigned __int64)v47);
  return a1;
}
