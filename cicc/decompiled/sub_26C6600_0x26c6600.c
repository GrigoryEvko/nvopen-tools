// Function: sub_26C6600
// Address: 0x26c6600
//
__int64 __fastcall sub_26C6600(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 *v13; // r13
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rdi
  __int64 v17[2]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-220h] BYREF
  _QWORD *v19; // [rsp+30h] [rbp-210h] BYREF
  _QWORD v20[4]; // [rsp+40h] [rbp-200h] BYREF
  void *v21; // [rsp+60h] [rbp-1E0h] BYREF
  int v22; // [rsp+68h] [rbp-1D8h]
  char v23; // [rsp+6Ch] [rbp-1D4h]
  __int64 v24; // [rsp+70h] [rbp-1D0h]
  __m128i v25; // [rsp+78h] [rbp-1C8h] BYREF
  __int64 v26; // [rsp+88h] [rbp-1B8h]
  __m128i v27; // [rsp+90h] [rbp-1B0h] BYREF
  __m128i v28; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned __int64 *v29; // [rsp+B0h] [rbp-190h] BYREF
  unsigned int v30; // [rsp+B8h] [rbp-188h]
  char v31; // [rsp+C0h] [rbp-180h] BYREF
  char v32; // [rsp+200h] [rbp-40h]
  int v33; // [rsp+204h] [rbp-3Ch]
  __int64 v34; // [rsp+208h] [rbp-38h]

  sub_B178C0((__int64)&v21, (__int64)"sample-profile-impl", (__int64)"AppliedSamples", 14, *(_QWORD *)a2);
  sub_B18290((__int64)&v21, "Applied ", 8u);
  sub_B16B10(v17, "NumSamples", 10, **(_QWORD **)(a2 + 8));
  sub_B826F0((__int64)&v21, (__int64)v17);
  if ( v19 != v20 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0(v17[0]);
  sub_B18290((__int64)&v21, " samples from profile (offset: ", 0x1Fu);
  sub_B169E0(v17, "LineOffset", 10, **(_DWORD **)(a2 + 16));
  sub_B826F0((__int64)&v21, (__int64)v17);
  if ( v19 != v20 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0(v17[0]);
  if ( **(_DWORD **)(a2 + 24) )
  {
    sub_B18290((__int64)&v21, ".", 1u);
    sub_B169E0(v17, "Discriminator", 13, **(_DWORD **)(a2 + 24));
    sub_B826F0((__int64)&v21, (__int64)v17);
    sub_2240A30((unsigned __int64 *)&v19);
    if ( (_QWORD *)v17[0] != v18 )
      j_j___libc_free_0(v17[0]);
  }
  sub_B18290((__int64)&v21, ")", 1u);
  v7 = _mm_loadu_si128(&v25);
  v8 = _mm_loadu_si128(&v27);
  v9 = _mm_loadu_si128(&v28);
  *(_DWORD *)(a1 + 8) = v22;
  v10 = v23;
  *(__m128i *)(a1 + 24) = v7;
  *(_BYTE *)(a1 + 12) = v10;
  v11 = v24;
  *(__m128i *)(a1 + 48) = v8;
  *(_QWORD *)(a1 + 16) = v11;
  *(__m128i *)(a1 + 64) = v9;
  v12 = v26;
  *(_QWORD *)a1 = &unk_49D9D40;
  *(_QWORD *)(a1 + 40) = v12;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  if ( v30 )
  {
    sub_26C5780(a1 + 80, (__int64)&v29, v3, v4, v5, v6);
    v21 = &unk_49D9D40;
    v15 = v29;
    *(_BYTE *)(a1 + 416) = v32;
    *(_DWORD *)(a1 + 420) = v33;
    *(_QWORD *)(a1 + 424) = v34;
    *(_QWORD *)a1 = &unk_49D9DE8;
    v13 = &v15[10 * v30];
    if ( v15 != v13 )
    {
      do
      {
        v13 -= 10;
        v16 = v13[4];
        if ( (unsigned __int64 *)v16 != v13 + 6 )
          j_j___libc_free_0(v16);
        if ( (unsigned __int64 *)*v13 != v13 + 2 )
          j_j___libc_free_0(*v13);
      }
      while ( v15 != v13 );
      v13 = v29;
    }
  }
  else
  {
    v13 = v29;
    *(_BYTE *)(a1 + 416) = v32;
    *(_DWORD *)(a1 + 420) = v33;
    *(_QWORD *)(a1 + 424) = v34;
    *(_QWORD *)a1 = &unk_49D9DE8;
  }
  if ( v13 != (unsigned __int64 *)&v31 )
    _libc_free((unsigned __int64)v13);
  return a1;
}
