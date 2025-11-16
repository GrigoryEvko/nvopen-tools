// Function: sub_1AFE3E0
// Address: 0x1afe3e0
//
void __fastcall sub_1AFE3E0(__int64 *a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // rax
  size_t v6; // rdx
  char *v7; // rsi
  _QWORD *v8; // r12
  _QWORD *v9; // r13
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  _QWORD *v12; // r13
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  void *v16; // [rsp+0h] [rbp-3F0h] BYREF
  int v17; // [rsp+8h] [rbp-3E8h]
  char v18; // [rsp+Ch] [rbp-3E4h]
  __int64 v19; // [rsp+10h] [rbp-3E0h]
  __m128i v20; // [rsp+18h] [rbp-3D8h]
  __int64 v21; // [rsp+28h] [rbp-3C8h]
  __int64 v22; // [rsp+30h] [rbp-3C0h]
  __m128i v23; // [rsp+38h] [rbp-3B8h]
  __int64 v24; // [rsp+48h] [rbp-3A8h]
  char v25; // [rsp+50h] [rbp-3A0h]
  _BYTE *v26; // [rsp+58h] [rbp-398h] BYREF
  __int64 v27; // [rsp+60h] [rbp-390h]
  _BYTE v28[352]; // [rsp+68h] [rbp-388h] BYREF
  char v29; // [rsp+1C8h] [rbp-228h]
  int v30; // [rsp+1CCh] [rbp-224h]
  __int64 v31; // [rsp+1D0h] [rbp-220h]
  void *v32; // [rsp+1E0h] [rbp-210h] BYREF
  int v33; // [rsp+1E8h] [rbp-208h]
  char v34; // [rsp+1ECh] [rbp-204h]
  __int64 v35; // [rsp+1F0h] [rbp-200h]
  __m128i v36; // [rsp+1F8h] [rbp-1F8h] BYREF
  __int64 v37; // [rsp+208h] [rbp-1E8h]
  __int64 v38; // [rsp+210h] [rbp-1E0h]
  __m128i v39; // [rsp+218h] [rbp-1D8h] BYREF
  __int64 v40; // [rsp+228h] [rbp-1C8h]
  char v41; // [rsp+230h] [rbp-1C0h]
  _BYTE *v42; // [rsp+238h] [rbp-1B8h] BYREF
  int v43; // [rsp+240h] [rbp-1B0h]
  _BYTE v44[352]; // [rsp+248h] [rbp-1A8h] BYREF
  char v45; // [rsp+3A8h] [rbp-48h]
  int v46; // [rsp+3ACh] [rbp-44h]
  __int64 v47; // [rsp+3B0h] [rbp-40h]

  v5 = sub_15E0530(*a1);
  if ( !sub_1602790(v5) )
  {
    v14 = sub_15E0530(*a1);
    v15 = sub_16033E0(v14);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL))(v15) )
      return;
  }
  sub_1AFE150((__int64)&v32, a2);
  sub_15CAB20((__int64)&v32, " with run-time trip count", 0x19u);
  v6 = 23;
  v7 = " without remainder loop";
  if ( *a3 != 2 )
  {
    v6 = 0;
    v7 = (char *)byte_3F871B3;
  }
  sub_15CAB20((__int64)&v32, v7, v6);
  v17 = v33;
  v20 = _mm_loadu_si128(&v36);
  v18 = v34;
  v23 = _mm_loadu_si128(&v39);
  v19 = v35;
  v21 = v37;
  v16 = &unk_49ECF68;
  v22 = v38;
  v25 = v41;
  if ( v41 )
    v24 = v40;
  v27 = 0x400000000LL;
  v26 = v28;
  if ( v43 )
  {
    sub_1AFDB00((__int64)&v26, (__int64)&v42);
    v12 = v42;
    v29 = v45;
    v30 = v46;
    v31 = v47;
    v16 = &unk_49ECF98;
    v32 = &unk_49ECF68;
    v8 = &v42[88 * v43];
    if ( v42 != (_BYTE *)v8 )
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
      v8 = v42;
      if ( v42 == v44 )
        goto LABEL_10;
      goto LABEL_9;
    }
  }
  else
  {
    v8 = v42;
    v29 = v45;
    v30 = v46;
    v31 = v47;
    v16 = &unk_49ECF98;
  }
  if ( v8 != (_QWORD *)v44 )
LABEL_9:
    _libc_free((unsigned __int64)v8);
LABEL_10:
  sub_143AA50(a1, (__int64)&v16);
  v9 = v26;
  v16 = &unk_49ECF68;
  v10 = &v26[88 * (unsigned int)v27];
  if ( v26 != (_BYTE *)v10 )
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
    v10 = v26;
  }
  if ( v10 != (_QWORD *)v28 )
    _libc_free((unsigned __int64)v10);
}
