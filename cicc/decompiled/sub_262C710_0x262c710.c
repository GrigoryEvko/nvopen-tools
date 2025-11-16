// Function: sub_262C710
// Address: 0x262c710
//
void __fastcall sub_262C710(__int64 a1, _BYTE *a2, __int64 a3, _QWORD *a4)
{
  char *v6; // rsi
  char *v7; // rax
  __m128i v8; // xmm1
  __m128i v9; // xmm0
  char *v10; // r15
  __int64 v11; // rax
  int *v12; // rdi
  int *v13; // rdx
  int *v14; // rax
  int *v15; // rdx
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  __int64 v19; // rax
  int *v20; // rcx
  int *v21; // rdx
  int *v22; // rax
  int *v23; // rdx
  __m128i v24; // xmm5
  char v26; // [rsp+1Fh] [rbp-171h] BYREF
  __m128i v27; // [rsp+20h] [rbp-170h] BYREF
  __m128i v28; // [rsp+30h] [rbp-160h] BYREF
  __int64 v29; // [rsp+40h] [rbp-150h]
  int v30; // [rsp+50h] [rbp-140h] BYREF
  int *v31; // [rsp+58h] [rbp-138h]
  int *v32; // [rsp+60h] [rbp-130h]
  int *v33; // [rsp+68h] [rbp-128h]
  __int64 v34; // [rsp+70h] [rbp-120h]
  __m128i v35; // [rsp+80h] [rbp-110h] BYREF
  __m128i v36; // [rsp+90h] [rbp-100h] BYREF
  __m128i v37; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v38; // [rsp+B0h] [rbp-E0h]
  int v39; // [rsp+C0h] [rbp-D0h] BYREF
  int *v40; // [rsp+C8h] [rbp-C8h]
  int *v41; // [rsp+D0h] [rbp-C0h]
  int *v42; // [rsp+D8h] [rbp-B8h]
  __int64 v43; // [rsp+E0h] [rbp-B0h]
  char *v44; // [rsp+F0h] [rbp-A0h] BYREF
  __m128i v45; // [rsp+F8h] [rbp-98h] BYREF
  __m128i v46; // [rsp+108h] [rbp-88h]
  __m128i v47; // [rsp+118h] [rbp-78h]
  __int64 v48; // [rsp+128h] [rbp-68h]
  int v49; // [rsp+138h] [rbp-58h] BYREF
  int *v50; // [rsp+140h] [rbp-50h]
  int *v51; // [rsp+148h] [rbp-48h]
  int *v52; // [rsp+150h] [rbp-40h]
  __int64 v53; // [rsp+158h] [rbp-38h]

  v27 = (__m128i)5uLL;
  v28.m128i_i64[0] = 0;
  v28.m128i_i8[8] = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = &v30;
  v33 = &v30;
  v34 = 0;
  if ( a2 )
  {
    v44 = &v45.m128i_i8[8];
    sub_2619AF0((__int64 *)&v44, a2, (__int64)&a2[a3]);
    v6 = v44;
  }
  else
  {
    v45.m128i_i8[8] = 0;
    v44 = &v45.m128i_i8[8];
    v6 = &v45.m128i_i8[8];
    v45.m128i_i64[0] = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         v6,
         1,
         0,
         &v26,
         &v35) )
  {
    sub_262C1A0(a1, (__int64)&v27);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v35.m128i_i64[0]);
  }
  if ( v44 != (char *)&v45.m128i_u64[1] )
    j_j___libc_free_0((unsigned __int64)v44);
  v7 = (char *)sub_B2F650((__int64)a2, a3);
  v8 = _mm_loadu_si128(&v27);
  v9 = _mm_loadu_si128(&v28);
  v10 = v7;
  v35.m128i_i64[0] = (__int64)a2;
  v35.m128i_i64[1] = a3;
  v38 = v29;
  v39 = 0;
  v40 = 0;
  v41 = &v39;
  v42 = &v39;
  v43 = 0;
  v36 = v8;
  v37 = v9;
  if ( v31 )
  {
    v11 = sub_261B2A0(v31, (__int64)&v39);
    v12 = (int *)v11;
    do
    {
      v13 = (int *)v11;
      v11 = *(_QWORD *)(v11 + 16);
    }
    while ( v11 );
    v41 = v13;
    v14 = v12;
    do
    {
      v15 = v14;
      v14 = (int *)*((_QWORD *)v14 + 3);
    }
    while ( v14 );
    v16 = _mm_loadu_si128(&v35);
    v42 = v15;
    v17 = _mm_loadu_si128(&v36);
    v18 = _mm_loadu_si128(&v37);
    v40 = v12;
    v43 = v34;
    v44 = v10;
    v48 = v38;
    v49 = 0;
    v50 = 0;
    v51 = &v49;
    v52 = &v49;
    v53 = 0;
    v45 = v16;
    v46 = v17;
    v47 = v18;
    if ( v12 )
    {
      v19 = sub_261B2A0(v12, (__int64)&v49);
      v20 = (int *)v19;
      do
      {
        v21 = (int *)v19;
        v19 = *(_QWORD *)(v19 + 16);
      }
      while ( v19 );
      v51 = v21;
      v22 = v20;
      do
      {
        v23 = v22;
        v22 = (int *)*((_QWORD *)v22 + 3);
      }
      while ( v22 );
      v52 = v23;
      v50 = v20;
      v53 = v43;
    }
  }
  else
  {
    v24 = _mm_loadu_si128(&v35);
    v48 = v29;
    v44 = v7;
    v49 = 0;
    v50 = 0;
    v51 = &v49;
    v52 = &v49;
    v53 = 0;
    v45 = v24;
    v46 = v8;
    v47 = v9;
  }
  sub_9CA630(a4, (__int64 *)&v44);
  sub_261C430(v50);
  sub_261C430(v40);
  sub_261C430(v31);
}
