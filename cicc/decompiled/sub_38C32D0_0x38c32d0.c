// Function: sub_38C32D0
// Address: 0x38c32d0
//
__int64 __fastcall sub_38C32D0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  _BYTE *v5; // rax
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rbx
  int *v12; // rdi
  int *v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  __m128i v17; // xmm0
  __m128i *v18; // rax
  __m128i *v19; // r15
  unsigned __int64 v20; // r13
  __int64 m128i_i64; // r8
  __int64 v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rax
  _QWORD *v25; // rdx
  char v26; // di
  _QWORD *v27; // rcx
  __int64 result; // rax
  __int64 *v29; // rax
  __int64 v30; // rdi
  char v31; // al
  unsigned __int64 v32; // rdi
  __int64 v34; // [rsp+8h] [rbp-E8h]
  __int64 v36; // [rsp+18h] [rbp-D8h]
  __int64 v37; // [rsp+20h] [rbp-D0h]
  int v39; // [rsp+30h] [rbp-C0h]
  __int64 v40; // [rsp+30h] [rbp-C0h]
  _QWORD *v41; // [rsp+30h] [rbp-C0h]
  __int64 v42; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v43; // [rsp+30h] [rbp-C0h]
  __m128i *v44; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-A8h]
  __m128i v46; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v47; // [rsp+60h] [rbp-90h] BYREF
  int v48; // [rsp+70h] [rbp-80h]
  __m128i *v49; // [rsp+80h] [rbp-70h] BYREF
  __int64 v50; // [rsp+88h] [rbp-68h]
  __m128i v51; // [rsp+90h] [rbp-60h] BYREF
  __m128i v52; // [rsp+A0h] [rbp-50h] BYREF
  int v53; // [rsp+B0h] [rbp-40h]
  __int64 v54; // [rsp+B8h] [rbp-38h]

  v5 = *(_BYTE **)(a2 + 184);
  v37 = (__int64)v5;
  v36 = 0;
  if ( v5 )
  {
    if ( (*v5 & 4) != 0 )
    {
      v29 = (__int64 *)*((_QWORD *)v5 - 1);
      v37 = (__int64)(v29 + 2);
      v36 = *v29;
    }
    else
    {
      v37 = 0;
    }
  }
  v6 = *(_BYTE **)(a2 + 152);
  v7 = *(_QWORD *)(a2 + 160);
  v39 = *(_DWORD *)(a2 + 176);
  if ( v6 )
  {
    v49 = &v51;
    sub_38BB9D0((__int64 *)&v49, v6, (__int64)&v6[v7]);
  }
  else
  {
    v50 = 0;
    v49 = &v51;
    v51.m128i_i8[0] = 0;
  }
  v52.m128i_i64[0] = v37;
  v52.m128i_i64[1] = v36;
  v53 = v39;
  v34 = a1 + 1200;
  v8 = sub_38C2D80(a1 + 1200, (__int64)&v49);
  v10 = v9;
  v11 = v8;
  if ( v8 == *(_QWORD *)(a1 + 1224) && v9 == a1 + 1208 )
  {
    sub_38BBFE0(*(_QWORD **)(a1 + 1216));
    *(_QWORD *)(a1 + 1224) = v10;
    *(_QWORD *)(a1 + 1216) = 0;
    *(_QWORD *)(a1 + 1232) = v10;
    *(_QWORD *)(a1 + 1240) = 0;
  }
  else if ( v9 != v8 )
  {
    do
    {
      v12 = (int *)v11;
      v11 = sub_220EF30(v11);
      v13 = sub_220F330(v12, (_QWORD *)(a1 + 1208));
      v14 = *((_QWORD *)v13 + 4);
      v15 = (unsigned __int64)v13;
      if ( (int *)v14 != v13 + 12 )
        j_j___libc_free_0(v14);
      j_j___libc_free_0(v15);
      --*(_QWORD *)(a1 + 1240);
    }
    while ( v10 != v11 );
  }
  if ( v49 != &v51 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( a3 )
  {
    v44 = &v46;
    sub_38BB9D0((__int64 *)&v44, a3, (__int64)&a3[a4]);
    v16 = v45;
    v47.m128i_i64[0] = v37;
    v47.m128i_i64[1] = v36;
    v48 = v39;
    v49 = &v51;
    if ( v44 != &v46 )
    {
      v49 = v44;
      v51.m128i_i64[0] = v46.m128i_i64[0];
      goto LABEL_16;
    }
  }
  else
  {
    v46.m128i_i8[0] = 0;
    v47.m128i_i64[0] = v37;
    v47.m128i_i64[1] = v36;
    v48 = v39;
    v49 = &v51;
    v16 = 0;
  }
  v51 = _mm_load_si128(&v46);
LABEL_16:
  v50 = v16;
  v17 = _mm_load_si128(&v47);
  v44 = &v46;
  v53 = v39;
  v45 = 0;
  v46.m128i_i8[0] = 0;
  v54 = a2;
  v52 = v17;
  v18 = (__m128i *)sub_22077B0(0x60u);
  v19 = v18 + 3;
  v20 = (unsigned __int64)v18;
  m128i_i64 = (__int64)v18[2].m128i_i64;
  v18[2].m128i_i64[0] = (__int64)v18[3].m128i_i64;
  if ( v49 == &v51 )
  {
    v18[3] = _mm_load_si128(&v51);
  }
  else
  {
    v18[2].m128i_i64[0] = (__int64)v49;
    v18[3].m128i_i64[0] = v51.m128i_i64[0];
  }
  v22 = v50;
  v23 = _mm_load_si128(&v52);
  v51.m128i_i8[0] = 0;
  v50 = 0;
  *(_QWORD *)(v20 + 40) = v22;
  *(__m128i *)(v20 + 64) = v23;
  v49 = &v51;
  v40 = m128i_i64;
  *(_DWORD *)(v20 + 80) = v53;
  *(_QWORD *)(v20 + 88) = v54;
  v24 = sub_38C30E0(v34, m128i_i64);
  if ( v25 )
  {
    v26 = 1;
    v27 = (_QWORD *)(a1 + 1208);
    if ( !v24 && v25 != v27 )
    {
      v30 = v40;
      v41 = v25;
      v31 = sub_38BC8E0(v30, (__int64)(v25 + 4));
      v27 = (_QWORD *)(a1 + 1208);
      v25 = v41;
      v26 = v31;
    }
    sub_220F040(v26, v20, v25, v27);
    ++*(_QWORD *)(a1 + 1240);
  }
  else
  {
    v32 = *(_QWORD *)(v20 + 32);
    if ( v19 != (__m128i *)v32 )
    {
      v42 = v24;
      j_j___libc_free_0(v32);
      v24 = v42;
    }
    v43 = v24;
    j_j___libc_free_0(v20);
    v20 = v43;
  }
  if ( v49 != &v51 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( v44 != &v46 )
    j_j___libc_free_0((unsigned __int64)v44);
  result = *(_QWORD *)(v20 + 40);
  *(_QWORD *)(a2 + 152) = *(_QWORD *)(v20 + 32);
  *(_QWORD *)(a2 + 160) = result;
  return result;
}
