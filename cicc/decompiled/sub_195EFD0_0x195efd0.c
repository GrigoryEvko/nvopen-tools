// Function: sub_195EFD0
// Address: 0x195efd0
//
void __fastcall sub_195EFD0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // r13
  _QWORD *v6; // r12
  _QWORD *v7; // rdi
  _QWORD *v8; // r13
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  void *v12; // [rsp+0h] [rbp-3F0h] BYREF
  int v13; // [rsp+8h] [rbp-3E8h]
  char v14; // [rsp+Ch] [rbp-3E4h]
  __int64 v15; // [rsp+10h] [rbp-3E0h]
  __m128i v16; // [rsp+18h] [rbp-3D8h]
  __int64 v17; // [rsp+28h] [rbp-3C8h]
  __int64 v18; // [rsp+30h] [rbp-3C0h]
  __m128i v19; // [rsp+38h] [rbp-3B8h]
  __int64 v20; // [rsp+48h] [rbp-3A8h]
  char v21; // [rsp+50h] [rbp-3A0h]
  _BYTE *v22; // [rsp+58h] [rbp-398h] BYREF
  __int64 v23; // [rsp+60h] [rbp-390h]
  _BYTE v24[352]; // [rsp+68h] [rbp-388h] BYREF
  char v25; // [rsp+1C8h] [rbp-228h]
  int v26; // [rsp+1CCh] [rbp-224h]
  __int64 v27; // [rsp+1D0h] [rbp-220h]
  void *v28; // [rsp+1E0h] [rbp-210h] BYREF
  int v29; // [rsp+1E8h] [rbp-208h]
  char v30; // [rsp+1ECh] [rbp-204h]
  __int64 v31; // [rsp+1F0h] [rbp-200h]
  __m128i v32; // [rsp+1F8h] [rbp-1F8h] BYREF
  __int64 v33; // [rsp+208h] [rbp-1E8h]
  __int64 v34; // [rsp+210h] [rbp-1E0h]
  __m128i v35; // [rsp+218h] [rbp-1D8h] BYREF
  __int64 v36; // [rsp+228h] [rbp-1C8h]
  char v37; // [rsp+230h] [rbp-1C0h]
  _BYTE *v38; // [rsp+238h] [rbp-1B8h] BYREF
  int v39; // [rsp+240h] [rbp-1B0h]
  _BYTE v40[352]; // [rsp+248h] [rbp-1A8h] BYREF
  char v41; // [rsp+3A8h] [rbp-48h]
  int v42; // [rsp+3ACh] [rbp-44h]
  __int64 v43; // [rsp+3B0h] [rbp-40h]

  v3 = sub_15E0530(*a1);
  if ( !sub_1602790(v3) )
  {
    v10 = sub_15E0530(*a1);
    v11 = sub_16033E0(v10);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v11 + 48LL))(v11) )
      return;
  }
  sub_15CA5C0((__int64)&v28, (__int64)"licm", (__int64)"LoadWithLoopInvariantAddressCondExecuted", 40, *a2);
  sub_15CAB20(
    (__int64)&v28,
    "failed to hoist load with loop-invariant address because load is conditionally executed",
    0x57u);
  v13 = v29;
  v16 = _mm_loadu_si128(&v32);
  v14 = v30;
  v19 = _mm_loadu_si128(&v35);
  v15 = v31;
  v17 = v33;
  v12 = &unk_49ECF68;
  v18 = v34;
  v21 = v37;
  if ( v37 )
    v20 = v36;
  v23 = 0x400000000LL;
  v22 = v24;
  if ( v39 )
  {
    sub_195ED40((__int64)&v22, (__int64)&v38);
    v8 = v38;
    v25 = v41;
    v26 = v42;
    v27 = v43;
    v12 = &unk_49ECFC8;
    v28 = &unk_49ECF68;
    v4 = &v38[88 * v39];
    if ( v38 != (_BYTE *)v4 )
    {
      do
      {
        v4 -= 11;
        v9 = (_QWORD *)v4[4];
        if ( v9 != v4 + 6 )
          j_j___libc_free_0(v9, v4[6] + 1LL);
        if ( (_QWORD *)*v4 != v4 + 2 )
          j_j___libc_free_0(*v4, v4[2] + 1LL);
      }
      while ( v8 != v4 );
      v4 = v38;
      if ( v38 == v40 )
        goto LABEL_8;
      goto LABEL_7;
    }
  }
  else
  {
    v4 = v38;
    v25 = v41;
    v26 = v42;
    v27 = v43;
    v12 = &unk_49ECFC8;
  }
  if ( v4 != (_QWORD *)v40 )
LABEL_7:
    _libc_free((unsigned __int64)v4);
LABEL_8:
  sub_143AA50(a1, (__int64)&v12);
  v5 = v22;
  v12 = &unk_49ECF68;
  v6 = &v22[88 * (unsigned int)v23];
  if ( v22 != (_BYTE *)v6 )
  {
    do
    {
      v6 -= 11;
      v7 = (_QWORD *)v6[4];
      if ( v7 != v6 + 6 )
        j_j___libc_free_0(v7, v6[6] + 1LL);
      if ( (_QWORD *)*v6 != v6 + 2 )
        j_j___libc_free_0(*v6, v6[2] + 1LL);
    }
    while ( v5 != v6 );
    v6 = v22;
  }
  if ( v6 != (_QWORD *)v24 )
    _libc_free((unsigned __int64)v6);
}
