// Function: sub_16CF050
// Address: 0x16cf050
//
__int64 __fastcall sub_16CF050(__int64 *a1, _QWORD *a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // rax
  char *v7; // r14
  __int64 v8; // rax
  size_t v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __m128i *v12; // rax
  _BYTE *v13; // rdi
  __int64 v14; // rdx
  size_t v15; // rsi
  __int64 v16; // r8
  __int64 v17; // rax
  _QWORD *v18; // rdi
  _QWORD *v19; // rdi
  unsigned int v20; // r12d
  size_t v22; // rdx
  __int64 v24; // [rsp+10h] [rbp-B0h]
  __int64 v26; // [rsp+20h] [rbp-A0h]
  _QWORD v27[2]; // [rsp+30h] [rbp-90h] BYREF
  char v28; // [rsp+40h] [rbp-80h]
  __int64 v29[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v30[2]; // [rsp+60h] [rbp-60h] BYREF
  _OWORD *v31; // [rsp+70h] [rbp-50h] BYREF
  size_t n; // [rsp+78h] [rbp-48h]
  _OWORD src[4]; // [rsp+80h] [rbp-40h] BYREF

  sub_2240AE0(a4, a2);
  LOWORD(src[0]) = 260;
  v31 = a4;
  sub_16C2DE0((__int64)v27, (__int64)&v31, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
  v6 = (a1[4] - a1[3]) >> 5;
  if ( (_DWORD)v6 )
  {
    v24 = 32LL * (unsigned int)v6;
    if ( (v28 & 1) != 0 )
    {
      v26 = 0;
      while ( 1 )
      {
        v7 = sub_16C44C0(2);
        v8 = a1[3] + v26;
        v29[0] = (__int64)v30;
        sub_16CD370(v29, *(_BYTE **)v8, *(_QWORD *)v8 + *(_QWORD *)(v8 + 8));
        v9 = strlen(v7);
        if ( v9 > 0x3FFFFFFFFFFFFFFFLL - v29[1] )
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490(v29, v7, v9, v10);
        v12 = (__m128i *)sub_2241490(v29, *a2, a2[1], v11);
        v31 = src;
        if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
        {
          src[0] = _mm_loadu_si128(v12 + 1);
        }
        else
        {
          v31 = (_OWORD *)v12->m128i_i64[0];
          *(_QWORD *)&src[0] = v12[1].m128i_i64[0];
        }
        n = v12->m128i_u64[1];
        v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
        v12->m128i_i64[1] = 0;
        v12[1].m128i_i8[0] = 0;
        v13 = (_BYTE *)*a4;
        if ( v31 == src )
        {
          v22 = n;
          if ( n )
          {
            if ( n == 1 )
              *v13 = src[0];
            else
              memcpy(v13, src, n);
            v22 = n;
            v13 = (_BYTE *)*a4;
          }
          a4[1] = v22;
          v13[v22] = 0;
          v13 = v31;
          goto LABEL_11;
        }
        v14 = *(_QWORD *)&src[0];
        v15 = n;
        if ( v13 == (_BYTE *)(a4 + 2) )
          break;
        v16 = a4[2];
        *a4 = v31;
        a4[1] = v15;
        a4[2] = v14;
        if ( !v13 )
          goto LABEL_37;
        v31 = v13;
        *(_QWORD *)&src[0] = v16;
LABEL_11:
        n = 0;
        *v13 = 0;
        if ( v31 != src )
          j_j___libc_free_0(v31, *(_QWORD *)&src[0] + 1LL);
        if ( (_QWORD *)v29[0] != v30 )
          j_j___libc_free_0(v29[0], v30[0] + 1LL);
        LOWORD(v30[0]) = 260;
        v29[0] = (__int64)a4;
        sub_16C2DE0((__int64)&v31, (__int64)v29, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
        if ( (v28 & 1) != 0 || !v27[0] )
        {
          if ( (src[0] & 1) == 0 )
            goto LABEL_17;
        }
        else
        {
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v27[0] + 8LL))(v27[0]);
          if ( (src[0] & 1) == 0 )
          {
LABEL_17:
            v28 &= ~1u;
            v27[0] = v31;
            goto LABEL_18;
          }
        }
        v26 += 32;
        v28 |= 1u;
        LODWORD(v27[0]) = (_DWORD)v31;
        v27[1] = n;
        if ( v26 == v24 )
          return 0;
      }
      *a4 = v31;
      a4[1] = v15;
      a4[2] = v14;
LABEL_37:
      v31 = src;
      v13 = src;
      goto LABEL_11;
    }
  }
  else
  {
    v20 = 0;
    if ( (v28 & 1) != 0 )
      return v20;
  }
LABEL_18:
  v17 = v27[0];
  v27[0] = 0;
  n = 0;
  v31 = (_OWORD *)v17;
  v18 = (_QWORD *)a1[1];
  *(_QWORD *)&src[0] = a3;
  if ( v18 == (_QWORD *)a1[2] )
  {
    sub_168C7C0(a1, (__int64)v18, (__int64)&v31);
    v19 = (_QWORD *)a1[1];
  }
  else
  {
    if ( v18 )
    {
      sub_16CE2D0(v18, &v31);
      v18 = (_QWORD *)a1[1];
    }
    v19 = v18 + 3;
    a1[1] = (__int64)v19;
  }
  v20 = -1431655765 * (((__int64)v19 - *a1) >> 3);
  sub_16CE300((__int64 *)&v31);
  if ( (v28 & 1) == 0 && v27[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v27[0] + 8LL))(v27[0]);
  return v20;
}
