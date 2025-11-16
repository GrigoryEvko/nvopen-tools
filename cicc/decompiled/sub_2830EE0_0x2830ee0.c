// Function: sub_2830EE0
// Address: 0x2830ee0
//
void __fastcall sub_2830EE0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rsi
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  unsigned __int64 *v18; // r15
  unsigned __int64 *v19; // r13
  unsigned __int64 *v20; // r12
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // r13
  __int64 v23; // r8
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-3A8h] BYREF
  __m128i v28; // [rsp+10h] [rbp-3A0h] BYREF
  void *v29; // [rsp+20h] [rbp-390h] BYREF
  int v30; // [rsp+28h] [rbp-388h]
  char v31; // [rsp+2Ch] [rbp-384h]
  __int64 v32; // [rsp+30h] [rbp-380h]
  __m128i v33; // [rsp+38h] [rbp-378h]
  __int64 v34; // [rsp+48h] [rbp-368h]
  __m128i v35; // [rsp+50h] [rbp-360h]
  __m128i v36; // [rsp+60h] [rbp-350h]
  unsigned __int64 *v37; // [rsp+70h] [rbp-340h] BYREF
  __int64 v38; // [rsp+78h] [rbp-338h]
  _BYTE v39[320]; // [rsp+80h] [rbp-330h] BYREF
  char v40; // [rsp+1C0h] [rbp-1F0h]
  int v41; // [rsp+1C4h] [rbp-1ECh]
  __int64 v42; // [rsp+1C8h] [rbp-1E8h]
  void *v43; // [rsp+1D0h] [rbp-1E0h] BYREF
  int v44; // [rsp+1D8h] [rbp-1D8h]
  char v45; // [rsp+1DCh] [rbp-1D4h]
  __int64 v46; // [rsp+1E0h] [rbp-1D0h]
  __m128i v47; // [rsp+1E8h] [rbp-1C8h] BYREF
  __int64 v48; // [rsp+1F8h] [rbp-1B8h]
  __m128i v49; // [rsp+200h] [rbp-1B0h] BYREF
  __m128i v50; // [rsp+210h] [rbp-1A0h] BYREF
  unsigned __int64 *v51; // [rsp+220h] [rbp-190h] BYREF
  unsigned int v52; // [rsp+228h] [rbp-188h]
  _BYTE v53[320]; // [rsp+230h] [rbp-180h] BYREF
  char v54; // [rsp+370h] [rbp-40h]
  int v55; // [rsp+374h] [rbp-3Ch]
  __int64 v56; // [rsp+378h] [rbp-38h]

  v3 = *a1;
  v4 = sub_B2BE50(*a1);
  if ( !sub_B6EA50(v4) )
  {
    v25 = sub_B2BE50(v3);
    v26 = sub_B6F970(v25);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v26 + 48LL))(v26) )
      return;
  }
  v9 = *a2;
  v10 = **(_QWORD **)(v9 + 32);
  sub_D4BD20(&v27, v9, v5, v6, v7, v8);
  sub_B157E0((__int64)&v28, &v27);
  sub_B17640((__int64)&v43, (__int64)"loop-interchange", (__int64)"InterchangeNotProfitable", 24, &v28, v10);
  sub_B18290((__int64)&v43, "Interchanging loops is not considered to improve cache locality nor vectorization.", 0x52u);
  v15 = _mm_loadu_si128(&v47);
  v16 = _mm_loadu_si128(&v49);
  v17 = _mm_loadu_si128(&v50);
  v37 = (unsigned __int64 *)v39;
  v30 = v44;
  v33 = v15;
  v31 = v45;
  v35 = v16;
  v32 = v46;
  v36 = v17;
  v29 = &unk_49D9D40;
  v34 = v48;
  v38 = 0x400000000LL;
  if ( v52 )
  {
    sub_2830C60((__int64)&v37, (__int64)&v51, v11, v12, v13, v14);
    v43 = &unk_49D9D40;
    v22 = v51;
    v40 = v54;
    v41 = v55;
    v42 = v56;
    v29 = &unk_49D9DB0;
    v23 = 10LL * v52;
    v18 = &v51[v23];
    if ( v51 != &v51[v23] )
    {
      do
      {
        v18 -= 10;
        v24 = v18[4];
        if ( (unsigned __int64 *)v24 != v18 + 6 )
          j_j___libc_free_0(v24);
        if ( (unsigned __int64 *)*v18 != v18 + 2 )
          j_j___libc_free_0(*v18);
      }
      while ( v22 != v18 );
      v18 = v51;
      if ( v51 == (unsigned __int64 *)v53 )
        goto LABEL_6;
      goto LABEL_5;
    }
  }
  else
  {
    v18 = v51;
    v40 = v54;
    v41 = v55;
    v42 = v56;
    v29 = &unk_49D9DB0;
  }
  if ( v18 != (unsigned __int64 *)v53 )
LABEL_5:
    _libc_free((unsigned __int64)v18);
LABEL_6:
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
  sub_1049740(a1, (__int64)&v29);
  v19 = v37;
  v29 = &unk_49D9D40;
  v20 = &v37[10 * (unsigned int)v38];
  if ( v37 != v20 )
  {
    do
    {
      v20 -= 10;
      v21 = v20[4];
      if ( (unsigned __int64 *)v21 != v20 + 6 )
        j_j___libc_free_0(v21);
      if ( (unsigned __int64 *)*v20 != v20 + 2 )
        j_j___libc_free_0(*v20);
    }
    while ( v19 != v20 );
    v20 = v37;
  }
  if ( v20 != (unsigned __int64 *)v39 )
    _libc_free((unsigned __int64)v20);
}
