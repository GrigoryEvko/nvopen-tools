// Function: sub_1BF4360
// Address: 0x1bf4360
//
void __fastcall sub_1BF4360(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r15
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  _QWORD *v11; // r12
  _QWORD *v12; // r13
  _QWORD *v13; // r12
  _QWORD *v14; // rdi
  _QWORD *v15; // r13
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19[2]; // [rsp+0h] [rbp-410h] BYREF
  void *v20; // [rsp+20h] [rbp-3F0h] BYREF
  int v21; // [rsp+28h] [rbp-3E8h]
  char v22; // [rsp+2Ch] [rbp-3E4h]
  __int64 v23; // [rsp+30h] [rbp-3E0h]
  __m128i v24; // [rsp+38h] [rbp-3D8h]
  __int64 v25; // [rsp+48h] [rbp-3C8h]
  __int64 v26; // [rsp+50h] [rbp-3C0h]
  __m128i v27; // [rsp+58h] [rbp-3B8h]
  __int64 v28; // [rsp+68h] [rbp-3A8h]
  char v29; // [rsp+70h] [rbp-3A0h]
  _BYTE *v30; // [rsp+78h] [rbp-398h] BYREF
  __int64 v31; // [rsp+80h] [rbp-390h]
  _BYTE v32[352]; // [rsp+88h] [rbp-388h] BYREF
  char v33; // [rsp+1E8h] [rbp-228h]
  int v34; // [rsp+1ECh] [rbp-224h]
  __int64 v35; // [rsp+1F0h] [rbp-220h]
  void *v36; // [rsp+200h] [rbp-210h] BYREF
  int v37; // [rsp+208h] [rbp-208h]
  char v38; // [rsp+20Ch] [rbp-204h]
  __int64 v39; // [rsp+210h] [rbp-200h]
  __m128i v40; // [rsp+218h] [rbp-1F8h] BYREF
  __int64 v41; // [rsp+228h] [rbp-1E8h]
  __int64 v42; // [rsp+230h] [rbp-1E0h]
  __m128i v43; // [rsp+238h] [rbp-1D8h] BYREF
  __int64 v44; // [rsp+248h] [rbp-1C8h]
  char v45; // [rsp+250h] [rbp-1C0h]
  _BYTE *v46; // [rsp+258h] [rbp-1B8h] BYREF
  int v47; // [rsp+260h] [rbp-1B0h]
  _BYTE v48[352]; // [rsp+268h] [rbp-1A8h] BYREF
  char v49; // [rsp+3C8h] [rbp-48h]
  int v50; // [rsp+3CCh] [rbp-44h]
  __int64 v51; // [rsp+3D0h] [rbp-40h]

  v6 = sub_15E0530(*a1);
  if ( !sub_1602790(v6) )
  {
    v17 = sub_15E0530(*a1);
    v18 = sub_16033E0(v17);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v18 + 48LL))(v18) )
      return;
  }
  v7 = *(_QWORD *)(a3 + 8);
  v8 = *(_QWORD *)(v7 + 40);
  sub_15C9090((__int64)v19, (_QWORD *)(v7 + 48));
  sub_15CA7C0((__int64)&v36, 11, *a2, (__int64)"CantReorderFPOps", 16, v19, v8);
  v36 = &unk_49F7190;
  sub_15CAB20((__int64)&v36, "loop not vectorized: cannot prove it is safe to reorder floating-point operations", 0x51u);
  v9 = _mm_loadu_si128(&v40);
  v10 = _mm_loadu_si128(&v43);
  v21 = v37;
  v24 = v9;
  v27 = v10;
  v22 = v38;
  v23 = v39;
  v25 = v41;
  v20 = &unk_49ECF68;
  v26 = v42;
  v29 = v45;
  if ( v45 )
    v28 = v44;
  v31 = 0x400000000LL;
  v30 = v32;
  if ( v47 )
  {
    sub_1BF40D0((__int64)&v30, (__int64)&v46);
    v15 = v46;
    v33 = v49;
    v34 = v50;
    v35 = v51;
    v20 = &unk_49F7190;
    v36 = &unk_49ECF68;
    v11 = &v46[88 * v47];
    if ( v46 != (_BYTE *)v11 )
    {
      do
      {
        v11 -= 11;
        v16 = (_QWORD *)v11[4];
        if ( v16 != v11 + 6 )
          j_j___libc_free_0(v16, v11[6] + 1LL);
        if ( (_QWORD *)*v11 != v11 + 2 )
          j_j___libc_free_0(*v11, v11[2] + 1LL);
      }
      while ( v15 != v11 );
      v11 = v46;
      if ( v46 == v48 )
        goto LABEL_8;
      goto LABEL_7;
    }
  }
  else
  {
    v33 = v49;
    v34 = v50;
    v35 = v51;
    v11 = v46;
    v20 = &unk_49F7190;
  }
  if ( v11 != (_QWORD *)v48 )
LABEL_7:
    _libc_free((unsigned __int64)v11);
LABEL_8:
  sub_143AA50(a1, (__int64)&v20);
  v12 = v30;
  v20 = &unk_49ECF68;
  v13 = &v30[88 * (unsigned int)v31];
  if ( v30 != (_BYTE *)v13 )
  {
    do
    {
      v13 -= 11;
      v14 = (_QWORD *)v13[4];
      if ( v14 != v13 + 6 )
        j_j___libc_free_0(v14, v13[6] + 1LL);
      if ( (_QWORD *)*v13 != v13 + 2 )
        j_j___libc_free_0(*v13, v13[2] + 1LL);
    }
    while ( v12 != v13 );
    v13 = v30;
  }
  if ( v13 != (_QWORD *)v32 )
    _libc_free((unsigned __int64)v13);
}
