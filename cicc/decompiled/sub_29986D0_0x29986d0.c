// Function: sub_29986D0
// Address: 0x29986d0
//
void __fastcall sub_29986D0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 *v9; // r12
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // r14
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  void *v17; // [rsp+0h] [rbp-390h] BYREF
  int v18; // [rsp+8h] [rbp-388h]
  char v19; // [rsp+Ch] [rbp-384h]
  __int64 v20; // [rsp+10h] [rbp-380h]
  __m128i v21; // [rsp+18h] [rbp-378h]
  __int64 v22; // [rsp+28h] [rbp-368h]
  __m128i v23; // [rsp+30h] [rbp-360h]
  __m128i v24; // [rsp+40h] [rbp-350h]
  unsigned __int64 *v25; // [rsp+50h] [rbp-340h] BYREF
  __int64 v26; // [rsp+58h] [rbp-338h]
  _BYTE v27[320]; // [rsp+60h] [rbp-330h] BYREF
  char v28; // [rsp+1A0h] [rbp-1F0h]
  int v29; // [rsp+1A4h] [rbp-1ECh]
  __int64 v30; // [rsp+1A8h] [rbp-1E8h]
  void *v31; // [rsp+1B0h] [rbp-1E0h] BYREF
  int v32; // [rsp+1B8h] [rbp-1D8h]
  char v33; // [rsp+1BCh] [rbp-1D4h]
  __int64 v34; // [rsp+1C0h] [rbp-1D0h]
  __m128i v35; // [rsp+1C8h] [rbp-1C8h] BYREF
  __int64 v36; // [rsp+1D8h] [rbp-1B8h]
  __m128i v37; // [rsp+1E0h] [rbp-1B0h] BYREF
  __m128i v38; // [rsp+1F0h] [rbp-1A0h] BYREF
  unsigned __int64 *v39; // [rsp+200h] [rbp-190h] BYREF
  unsigned int v40; // [rsp+208h] [rbp-188h]
  _BYTE v41[320]; // [rsp+210h] [rbp-180h] BYREF
  char v42; // [rsp+350h] [rbp-40h]
  int v43; // [rsp+354h] [rbp-3Ch]
  __int64 v44; // [rsp+358h] [rbp-38h]

  v3 = *a1;
  v4 = sub_B2BE50(*a1);
  if ( !sub_B6EA50(v4) )
  {
    v15 = sub_B2BE50(v3);
    v16 = sub_B6F970(v15);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v16 + 48LL))(v16) )
      return;
  }
  sub_B174A0((__int64)&v31, (__int64)"tailcallelim", (__int64)"tailcall-recursion", 18, *a2);
  sub_B18290((__int64)&v31, "transforming tail recursion into loop", 0x25u);
  v25 = (unsigned __int64 *)v27;
  v18 = v32;
  v17 = &unk_49D9D40;
  v19 = v33;
  v21 = _mm_loadu_si128(&v35);
  v20 = v34;
  v23 = _mm_loadu_si128(&v37);
  v22 = v36;
  v26 = 0x400000000LL;
  v24 = _mm_loadu_si128(&v38);
  if ( v40 )
  {
    sub_2998450((__int64)&v25, (__int64)&v39, v5, v6, v7, v8);
    v31 = &unk_49D9D40;
    v13 = v39;
    v28 = v42;
    v29 = v43;
    v30 = v44;
    v17 = &unk_49D9D78;
    v9 = &v39[10 * v40];
    if ( v39 != v9 )
    {
      do
      {
        v9 -= 10;
        v14 = v9[4];
        if ( (unsigned __int64 *)v14 != v9 + 6 )
          j_j___libc_free_0(v14);
        if ( (unsigned __int64 *)*v9 != v9 + 2 )
          j_j___libc_free_0(*v9);
      }
      while ( v13 != v9 );
      v9 = v39;
      if ( v39 == (unsigned __int64 *)v41 )
        goto LABEL_6;
      goto LABEL_5;
    }
  }
  else
  {
    v9 = v39;
    v28 = v42;
    v29 = v43;
    v30 = v44;
    v17 = &unk_49D9D78;
  }
  if ( v9 != (unsigned __int64 *)v41 )
LABEL_5:
    _libc_free((unsigned __int64)v9);
LABEL_6:
  sub_1049740(a1, (__int64)&v17);
  v10 = v25;
  v17 = &unk_49D9D40;
  v11 = &v25[10 * (unsigned int)v26];
  if ( v25 != v11 )
  {
    do
    {
      v11 -= 10;
      v12 = v11[4];
      if ( (unsigned __int64 *)v12 != v11 + 6 )
        j_j___libc_free_0(v12);
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        j_j___libc_free_0(*v11);
    }
    while ( v10 != v11 );
    v11 = v25;
  }
  if ( v11 != (unsigned __int64 *)v27 )
    _libc_free((unsigned __int64)v11);
}
