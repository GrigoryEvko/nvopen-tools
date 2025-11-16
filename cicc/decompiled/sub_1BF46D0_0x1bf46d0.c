// Function: sub_1BF46D0
// Address: 0x1bf46d0
//
void __fastcall sub_1BF46D0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rax
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  _QWORD *v8; // rbx
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  _QWORD *v12; // r12
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-428h]
  __int64 v17; // [rsp+18h] [rbp-418h] BYREF
  __m128i v18[2]; // [rsp+20h] [rbp-410h] BYREF
  void *v19; // [rsp+40h] [rbp-3F0h] BYREF
  int v20; // [rsp+48h] [rbp-3E8h]
  char v21; // [rsp+4Ch] [rbp-3E4h]
  __int64 v22; // [rsp+50h] [rbp-3E0h]
  __m128i v23; // [rsp+58h] [rbp-3D8h]
  __int64 v24; // [rsp+68h] [rbp-3C8h]
  __int64 v25; // [rsp+70h] [rbp-3C0h]
  __m128i v26; // [rsp+78h] [rbp-3B8h]
  __int64 v27; // [rsp+88h] [rbp-3A8h]
  char v28; // [rsp+90h] [rbp-3A0h]
  _BYTE *v29; // [rsp+98h] [rbp-398h] BYREF
  __int64 v30; // [rsp+A0h] [rbp-390h]
  _BYTE v31[352]; // [rsp+A8h] [rbp-388h] BYREF
  char v32; // [rsp+208h] [rbp-228h]
  int v33; // [rsp+20Ch] [rbp-224h]
  __int64 v34; // [rsp+210h] [rbp-220h]
  void *v35; // [rsp+220h] [rbp-210h] BYREF
  int v36; // [rsp+228h] [rbp-208h]
  char v37; // [rsp+22Ch] [rbp-204h]
  __int64 v38; // [rsp+230h] [rbp-200h]
  __m128i v39; // [rsp+238h] [rbp-1F8h] BYREF
  __int64 v40; // [rsp+248h] [rbp-1E8h]
  __int64 v41; // [rsp+250h] [rbp-1E0h]
  __m128i v42; // [rsp+258h] [rbp-1D8h] BYREF
  __int64 v43; // [rsp+268h] [rbp-1C8h]
  char v44; // [rsp+270h] [rbp-1C0h]
  _BYTE *v45; // [rsp+278h] [rbp-1B8h] BYREF
  int v46; // [rsp+280h] [rbp-1B0h]
  _BYTE v47[352]; // [rsp+288h] [rbp-1A8h] BYREF
  char v48; // [rsp+3E8h] [rbp-48h]
  int v49; // [rsp+3ECh] [rbp-44h]
  __int64 v50; // [rsp+3F0h] [rbp-40h]

  v5 = sub_15E0530(*a1);
  if ( !sub_1602790(v5) )
  {
    v14 = sub_15E0530(*a1);
    v15 = sub_16033E0(v14);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL))(v15) )
      return;
  }
  v16 = **(_QWORD **)(*a3 + 32);
  sub_13FD840(&v17, *a3);
  sub_15C9090((__int64)v18, &v17);
  sub_15CA7C0((__int64)&v35, 12, *a2, (__int64)"CantReorderMemOps", 17, v18, v16);
  v35 = &unk_49F71C0;
  sub_15CAB20((__int64)&v35, "loop not vectorized: cannot prove it is safe to reorder memory operations", 0x49u);
  v6 = _mm_loadu_si128(&v39);
  v7 = _mm_loadu_si128(&v42);
  v20 = v36;
  v23 = v6;
  v21 = v37;
  v26 = v7;
  v22 = v38;
  v24 = v40;
  v19 = &unk_49ECF68;
  v25 = v41;
  v28 = v44;
  if ( v44 )
    v27 = v43;
  v30 = 0x400000000LL;
  v29 = v31;
  if ( v46 )
  {
    sub_1BF40D0((__int64)&v29, (__int64)&v45);
    v32 = v48;
    v33 = v49;
    v34 = v50;
    v12 = v45;
    v19 = &unk_49F71C0;
    v35 = &unk_49ECF68;
    v8 = &v45[88 * v46];
    if ( v45 != (_BYTE *)v8 )
    {
      do
      {
        v8 -= 11;
        v13 = (_QWORD *)v8[4];
        if ( v13 != v8 + 6 )
          j_j___libc_free_0(v13, v8[6] + 1LL);
        if ( (_QWORD *)*v8 != v8 + 2 )
          j_j___libc_free_0(*v8, v8[2] + 1LL);
      }
      while ( v12 != v8 );
      v8 = v45;
      if ( v45 == v47 )
        goto LABEL_8;
      goto LABEL_7;
    }
  }
  else
  {
    v8 = v45;
    v32 = v48;
    v33 = v49;
    v34 = v50;
    v19 = &unk_49F71C0;
  }
  if ( v8 != (_QWORD *)v47 )
LABEL_7:
    _libc_free((unsigned __int64)v8);
LABEL_8:
  if ( v17 )
    sub_161E7C0((__int64)&v17, v17);
  sub_143AA50(a1, (__int64)&v19);
  v9 = v29;
  v19 = &unk_49ECF68;
  v10 = &v29[88 * (unsigned int)v30];
  if ( v29 != (_BYTE *)v10 )
  {
    do
    {
      v10 -= 11;
      v11 = (_QWORD *)v10[4];
      if ( v11 != v10 + 6 )
        j_j___libc_free_0(v11, v10[6] + 1LL);
      if ( (_QWORD *)*v10 != v10 + 2 )
        j_j___libc_free_0(*v10, v10[2] + 1LL);
    }
    while ( v9 != v10 );
    v10 = v29;
  }
  if ( v10 != (_QWORD *)v31 )
    _libc_free((unsigned __int64)v10);
}
