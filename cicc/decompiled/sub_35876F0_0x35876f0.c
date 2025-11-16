// Function: sub_35876F0
// Address: 0x35876f0
//
__int64 __fastcall sub_35876F0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __m128i v6; // xmm0
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  char v15; // al
  __int64 v16; // rax
  const char *v17; // rax
  unsigned __int64 *v18; // r13
  unsigned __int64 *v20; // rbx
  unsigned __int64 v21; // rdi
  __m128i v22; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v23[2]; // [rsp+20h] [rbp-220h] BYREF
  _QWORD *v24; // [rsp+30h] [rbp-210h]
  _QWORD v25[4]; // [rsp+40h] [rbp-200h] BYREF
  void *v26; // [rsp+60h] [rbp-1E0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-1D8h]
  __int64 v28; // [rsp+70h] [rbp-1D0h]
  __m128i v29; // [rsp+78h] [rbp-1C8h] BYREF
  const char *v30; // [rsp+88h] [rbp-1B8h]
  __m128i v31; // [rsp+90h] [rbp-1B0h] BYREF
  __m128i v32; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned __int64 *v33; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v34; // [rsp+B8h] [rbp-188h]
  _BYTE v35[320]; // [rsp+C0h] [rbp-180h] BYREF
  char v36; // [rsp+200h] [rbp-40h]
  int v37; // [rsp+204h] [rbp-3Ch]
  __int64 v38; // [rsp+208h] [rbp-38h]

  v4 = *a2;
  v5 = *(_QWORD *)(v4 + 24);
  sub_B157E0((__int64)&v22, (_QWORD *)(v4 + 56));
  v6 = _mm_loadu_si128(&v22);
  v7 = **(_QWORD **)(v5 + 32);
  v27 = 0x200000015LL;
  v29 = v6;
  v28 = v7;
  v30 = "sample-profile-impl";
  v31.m128i_i64[0] = (__int64)"AppliedSamples";
  v33 = (unsigned __int64 *)v35;
  v34 = 0x400000000LL;
  v38 = v5;
  v32.m128i_i8[8] = 0;
  v26 = &unk_4A28EB8;
  v31.m128i_i64[1] = 14;
  v36 = 0;
  v37 = -1;
  sub_B18290((__int64)&v26, "Applied ", 8u);
  sub_B16B10(v22.m128i_i64, "NumSamples", 10, *(_QWORD *)a2[1]);
  sub_3584180((__int64)&v26, (__int64)&v22);
  if ( v24 != v25 )
    j_j___libc_free_0((unsigned __int64)v24);
  if ( (_QWORD *)v22.m128i_i64[0] != v23 )
    j_j___libc_free_0(v22.m128i_u64[0]);
  sub_B18290((__int64)&v26, " samples from profile (offset: ", 0x1Fu);
  sub_B169E0(v22.m128i_i64, "LineOffset", 10, *(_DWORD *)a2[2]);
  sub_3584180((__int64)&v26, (__int64)&v22);
  if ( v24 != v25 )
    j_j___libc_free_0((unsigned __int64)v24);
  if ( (_QWORD *)v22.m128i_i64[0] != v23 )
    j_j___libc_free_0(v22.m128i_u64[0]);
  if ( *(_DWORD *)a2[3] )
  {
    sub_B18290((__int64)&v26, ".", 1u);
    sub_B169E0(v22.m128i_i64, "Discriminator", 13, *(_DWORD *)a2[3]);
    sub_3584180((__int64)&v26, (__int64)&v22);
    if ( v24 != v25 )
      j_j___libc_free_0((unsigned __int64)v24);
    if ( (_QWORD *)v22.m128i_i64[0] != v23 )
      j_j___libc_free_0(v22.m128i_u64[0]);
  }
  sub_B18290((__int64)&v26, ")", 1u);
  v12 = _mm_loadu_si128(&v29);
  v13 = _mm_loadu_si128(&v31);
  v14 = _mm_loadu_si128(&v32);
  *(_DWORD *)(a1 + 8) = v27;
  v15 = BYTE4(v27);
  *(__m128i *)(a1 + 24) = v12;
  *(_BYTE *)(a1 + 12) = v15;
  v16 = v28;
  *(__m128i *)(a1 + 48) = v13;
  *(_QWORD *)(a1 + 16) = v16;
  *(__m128i *)(a1 + 64) = v14;
  v17 = v30;
  *(_QWORD *)a1 = &unk_49D9D40;
  *(_QWORD *)(a1 + 40) = v17;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  if ( (_DWORD)v34 )
  {
    sub_3586750(a1 + 80, (__int64)&v33, v8, v9, v10, v11);
    v26 = &unk_49D9D40;
    v20 = v33;
    *(_BYTE *)(a1 + 416) = v36;
    *(_DWORD *)(a1 + 420) = v37;
    *(_QWORD *)(a1 + 424) = v38;
    *(_QWORD *)a1 = &unk_4A28EB8;
    v18 = &v20[10 * (unsigned int)v34];
    if ( v20 != v18 )
    {
      do
      {
        v18 -= 10;
        v21 = v18[4];
        if ( (unsigned __int64 *)v21 != v18 + 6 )
          j_j___libc_free_0(v21);
        if ( (unsigned __int64 *)*v18 != v18 + 2 )
          j_j___libc_free_0(*v18);
      }
      while ( v20 != v18 );
      v18 = v33;
    }
  }
  else
  {
    v18 = v33;
    *(_BYTE *)(a1 + 416) = v36;
    *(_DWORD *)(a1 + 420) = v37;
    *(_QWORD *)(a1 + 424) = v38;
    *(_QWORD *)a1 = &unk_4A28EB8;
  }
  if ( v18 != (unsigned __int64 *)v35 )
    _libc_free((unsigned __int64)v18);
  return a1;
}
