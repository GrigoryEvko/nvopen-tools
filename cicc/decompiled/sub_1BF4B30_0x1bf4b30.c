// Function: sub_1BF4B30
// Address: 0x1bf4b30
//
void __fastcall sub_1BF4B30(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  char *v7; // rax
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  _QWORD *v10; // rbx
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  _QWORD *v13; // rdi
  _QWORD *v14; // r12
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-418h] BYREF
  __m128i v19[2]; // [rsp+10h] [rbp-410h] BYREF
  void *v20; // [rsp+30h] [rbp-3F0h] BYREF
  int v21; // [rsp+38h] [rbp-3E8h]
  char v22; // [rsp+3Ch] [rbp-3E4h]
  __int64 v23; // [rsp+40h] [rbp-3E0h]
  __m128i v24; // [rsp+48h] [rbp-3D8h]
  __int64 v25; // [rsp+58h] [rbp-3C8h]
  __int64 v26; // [rsp+60h] [rbp-3C0h]
  __m128i v27; // [rsp+68h] [rbp-3B8h]
  __int64 v28; // [rsp+78h] [rbp-3A8h]
  char v29; // [rsp+80h] [rbp-3A0h]
  _BYTE *v30; // [rsp+88h] [rbp-398h] BYREF
  __int64 v31; // [rsp+90h] [rbp-390h]
  _BYTE v32[352]; // [rsp+98h] [rbp-388h] BYREF
  char v33; // [rsp+1F8h] [rbp-228h]
  int v34; // [rsp+1FCh] [rbp-224h]
  __int64 v35; // [rsp+200h] [rbp-220h]
  void *v36; // [rsp+210h] [rbp-210h] BYREF
  int v37; // [rsp+218h] [rbp-208h]
  char v38; // [rsp+21Ch] [rbp-204h]
  __int64 v39; // [rsp+220h] [rbp-200h]
  __m128i v40; // [rsp+228h] [rbp-1F8h] BYREF
  __int64 v41; // [rsp+238h] [rbp-1E8h]
  __int64 v42; // [rsp+240h] [rbp-1E0h]
  __m128i v43; // [rsp+248h] [rbp-1D8h] BYREF
  __int64 v44; // [rsp+258h] [rbp-1C8h]
  char v45; // [rsp+260h] [rbp-1C0h]
  _BYTE *v46; // [rsp+268h] [rbp-1B8h] BYREF
  int v47; // [rsp+270h] [rbp-1B0h]
  _BYTE v48[352]; // [rsp+278h] [rbp-1A8h] BYREF
  char v49; // [rsp+3D8h] [rbp-48h]
  int v50; // [rsp+3DCh] [rbp-44h]
  __int64 v51; // [rsp+3E0h] [rbp-40h]

  v5 = sub_15E0530(*a1);
  if ( !sub_1602790(v5) )
  {
    v16 = sub_15E0530(*a1);
    v17 = sub_16033E0(v16);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v17 + 48LL))(v17) )
      return;
  }
  v6 = **(_QWORD **)(*a3 + 32);
  sub_13FD840(&v18, *a3);
  sub_15C9090((__int64)v19, &v18);
  v7 = sub_1BF18B0(a2);
  sub_15CA680((__int64)&v36, (__int64)v7, (__int64)"AllDisabled", 11, v19, v6);
  sub_15CAB20(
    (__int64)&v36,
    "loop not vectorized: vectorization and interleaving are explicitly disabled, or the loop has already been vectorized",
    0x74u);
  v8 = _mm_loadu_si128(&v40);
  v9 = _mm_loadu_si128(&v43);
  v21 = v37;
  v24 = v8;
  v22 = v38;
  v27 = v9;
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
    v14 = v46;
    v33 = v49;
    v34 = v50;
    v35 = v51;
    v20 = &unk_49ECFF8;
    v36 = &unk_49ECF68;
    v10 = &v46[88 * v47];
    if ( v46 != (_BYTE *)v10 )
    {
      do
      {
        v10 -= 11;
        v15 = (_QWORD *)v10[4];
        if ( v15 != v10 + 6 )
          j_j___libc_free_0(v15, v10[6] + 1LL);
        if ( (_QWORD *)*v10 != v10 + 2 )
          j_j___libc_free_0(*v10, v10[2] + 1LL);
      }
      while ( v14 != v10 );
      v10 = v46;
      if ( v46 == v48 )
        goto LABEL_8;
      goto LABEL_7;
    }
  }
  else
  {
    v10 = v46;
    v33 = v49;
    v34 = v50;
    v35 = v51;
    v20 = &unk_49ECFF8;
  }
  if ( v10 != (_QWORD *)v48 )
LABEL_7:
    _libc_free((unsigned __int64)v10);
LABEL_8:
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  sub_143AA50(a1, (__int64)&v20);
  v11 = v30;
  v20 = &unk_49ECF68;
  v12 = &v30[88 * (unsigned int)v31];
  if ( v30 != (_BYTE *)v12 )
  {
    do
    {
      v12 -= 11;
      v13 = (_QWORD *)v12[4];
      if ( v13 != v12 + 6 )
        j_j___libc_free_0(v13, v12[6] + 1LL);
      if ( (_QWORD *)*v12 != v12 + 2 )
        j_j___libc_free_0(*v12, v12[2] + 1LL);
    }
    while ( v11 != v12 );
    v12 = v30;
  }
  if ( v12 != (_QWORD *)v32 )
    _libc_free((unsigned __int64)v12);
}
