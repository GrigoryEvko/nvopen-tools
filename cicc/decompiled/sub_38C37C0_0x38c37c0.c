// Function: sub_38C37C0
// Address: 0x38c37c0
//
__int64 __fastcall sub_38C37C0(__int64 a1, __int64 a2, int a3, int a4, int a5, _BYTE *a6, int a7, __int64 a8)
{
  const char *v8; // r15
  __int64 v9; // r13
  __int64 v12; // rax
  __m128i v13; // xmm0
  __m128i *v14; // rax
  __m128i *v15; // rcx
  unsigned __int64 v16; // r13
  __int64 m128i_i64; // r8
  __int64 v18; // rax
  __m128i v19; // xmm1
  __int64 v20; // rax
  _QWORD *v21; // rdx
  char v22; // di
  _QWORD *v23; // rcx
  unsigned __int8 v24; // r9
  __int64 result; // rax
  _QWORD *v26; // r15
  char v27; // al
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v31; // [rsp+8h] [rbp-E8h]
  __m128i *v32; // [rsp+10h] [rbp-E0h]
  char v33; // [rsp+10h] [rbp-E0h]
  _QWORD *v34; // [rsp+10h] [rbp-E0h]
  __int64 v35; // [rsp+10h] [rbp-E0h]
  __int64 v36; // [rsp+10h] [rbp-E0h]
  _BYTE *v39[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+30h] [rbp-C0h] BYREF
  __m128i *v41; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+48h] [rbp-A8h]
  __m128i v43; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v44; // [rsp+60h] [rbp-90h] BYREF
  int v45; // [rsp+70h] [rbp-80h]
  __m128i *v46; // [rsp+80h] [rbp-70h]
  __int64 v47; // [rsp+88h] [rbp-68h]
  __m128i v48; // [rsp+90h] [rbp-60h] BYREF
  __m128i v49; // [rsp+A0h] [rbp-50h] BYREF
  int v50; // [rsp+B0h] [rbp-40h]
  __int64 v51; // [rsp+B8h] [rbp-38h]

  v8 = byte_3F871B3;
  v9 = 0;
  if ( a6 )
  {
    v8 = 0;
    if ( (*a6 & 4) != 0 )
    {
      v26 = (_QWORD *)*((_QWORD *)a6 - 1);
      v9 = *v26;
      v8 = (const char *)(v26 + 2);
    }
  }
  sub_16E2FC0((__int64 *)v39, a2);
  if ( v39[0] )
  {
    v41 = &v43;
    sub_38BB9D0((__int64 *)&v41, v39[0], (__int64)&v39[0][(unsigned __int64)v39[1]]);
    v44.m128i_i64[0] = (__int64)v8;
    v44.m128i_i64[1] = v9;
    v12 = v42;
    v45 = a7;
    v46 = &v48;
    if ( v41 != &v43 )
    {
      v46 = v41;
      v48.m128i_i64[0] = v43.m128i_i64[0];
      goto LABEL_7;
    }
  }
  else
  {
    v44.m128i_i64[0] = (__int64)v8;
    v43.m128i_i8[0] = 0;
    v44.m128i_i64[1] = v9;
    v46 = &v48;
    v45 = a7;
    v12 = 0;
  }
  v48 = _mm_load_si128(&v43);
LABEL_7:
  v47 = v12;
  v13 = _mm_load_si128(&v44);
  v41 = &v43;
  v42 = 0;
  v43.m128i_i8[0] = 0;
  v50 = a7;
  v51 = 0;
  v49 = v13;
  v14 = (__m128i *)sub_22077B0(0x60u);
  v15 = v14 + 3;
  v16 = (unsigned __int64)v14;
  m128i_i64 = (__int64)v14[2].m128i_i64;
  v14[2].m128i_i64[0] = (__int64)v14[3].m128i_i64;
  if ( v46 == &v48 )
  {
    v14[3] = _mm_load_si128(&v48);
  }
  else
  {
    v14[2].m128i_i64[0] = (__int64)v46;
    v14[3].m128i_i64[0] = v48.m128i_i64[0];
  }
  v18 = v47;
  v19 = _mm_load_si128(&v49);
  *(_QWORD *)(v16 + 88) = 0;
  *(_QWORD *)(v16 + 40) = v18;
  LODWORD(v18) = v50;
  *(__m128i *)(v16 + 64) = v19;
  *(_DWORD *)(v16 + 80) = v18;
  v32 = v15;
  v46 = &v48;
  v47 = 0;
  v48.m128i_i8[0] = 0;
  v31 = m128i_i64;
  v20 = sub_38C30E0(a1 + 1200, m128i_i64);
  if ( v21 )
  {
    v22 = 1;
    v23 = (_QWORD *)(a1 + 1208);
    if ( !v20 && v23 != v21 )
    {
      v34 = v21;
      v27 = sub_38BC8E0(v31, (__int64)(v21 + 4));
      v23 = (_QWORD *)(a1 + 1208);
      v21 = v34;
      v22 = v27;
    }
    sub_220F040(v22, v16, v21, v23);
    v33 = 1;
    ++*(_QWORD *)(a1 + 1240);
  }
  else
  {
    v28 = *(_QWORD *)(v16 + 32);
    if ( v32 != (__m128i *)v28 )
    {
      v35 = v20;
      j_j___libc_free_0(v28);
      v20 = v35;
    }
    v36 = v20;
    j_j___libc_free_0(v16);
    v29 = v36;
    v33 = 0;
    v16 = v29;
  }
  if ( v46 != &v48 )
    j_j___libc_free_0((unsigned __int64)v46);
  if ( v41 != &v43 )
    j_j___libc_free_0((unsigned __int64)v41);
  if ( (__int64 *)v39[0] != &v40 )
    j_j___libc_free_0((unsigned __int64)v39[0]);
  if ( !v33 )
    return *(_QWORD *)(v16 + 88);
  v24 = 2;
  if ( (a4 & 0x20000000) == 0 )
    v24 = (a4 & 4) == 0 ? 3 : 1;
  result = sub_38BE590(a1, *(unsigned __int8 **)(v16 + 32), *(_QWORD *)(v16 + 40), a3, a4, v24, a5, (__int64)a6, a7, a8);
  *(_QWORD *)(v16 + 88) = result;
  return result;
}
