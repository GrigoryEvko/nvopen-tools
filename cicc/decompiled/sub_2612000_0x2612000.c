// Function: sub_2612000
// Address: 0x2612000
//
__int64 __fastcall sub_2612000(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rax
  __m128i *v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // eax
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 *v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // r13
  unsigned __int64 v18; // rdi
  __int64 v20; // rsi
  unsigned __int64 *v21; // r13
  unsigned __int64 *v22; // r15
  _BYTE *v23[2]; // [rsp+10h] [rbp-2D0h] BYREF
  __int64 v24; // [rsp+20h] [rbp-2C0h] BYREF
  __int64 *v25; // [rsp+30h] [rbp-2B0h]
  __int64 v26; // [rsp+38h] [rbp-2A8h]
  __int64 v27; // [rsp+40h] [rbp-2A0h] BYREF
  __m128i v28; // [rsp+50h] [rbp-290h] BYREF
  _BYTE *v29[2]; // [rsp+60h] [rbp-280h] BYREF
  __int64 v30; // [rsp+70h] [rbp-270h] BYREF
  __int64 *v31; // [rsp+80h] [rbp-260h]
  __int64 v32; // [rsp+88h] [rbp-258h]
  __int64 v33; // [rsp+90h] [rbp-250h] BYREF
  __m128i v34; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v35[2]; // [rsp+B0h] [rbp-230h] BYREF
  _QWORD v36[2]; // [rsp+C0h] [rbp-220h] BYREF
  __int64 v37[2]; // [rsp+D0h] [rbp-210h] BYREF
  _QWORD v38[2]; // [rsp+E0h] [rbp-200h] BYREF
  __m128i v39; // [rsp+F0h] [rbp-1F0h]
  void *v40; // [rsp+100h] [rbp-1E0h] BYREF
  int v41; // [rsp+108h] [rbp-1D8h]
  char v42; // [rsp+10Ch] [rbp-1D4h]
  __int64 v43; // [rsp+110h] [rbp-1D0h]
  __m128i v44; // [rsp+118h] [rbp-1C8h] BYREF
  __int64 v45; // [rsp+128h] [rbp-1B8h]
  __m128i v46; // [rsp+130h] [rbp-1B0h] BYREF
  __m128i v47; // [rsp+140h] [rbp-1A0h] BYREF
  unsigned __int64 *v48; // [rsp+150h] [rbp-190h]
  unsigned int v49; // [rsp+158h] [rbp-188h]
  _BYTE v50[324]; // [rsp+160h] [rbp-180h] BYREF
  int v51; // [rsp+2A4h] [rbp-3Ch]
  __int64 v52; // [rsp+2A8h] [rbp-38h]

  sub_B176B0((__int64)&v40, (__int64)"inline", (__int64)"NoDefinition", 12, *(_QWORD *)a2);
  sub_B16080((__int64)v23, "Callee", 6, **(unsigned __int8 ***)(a2 + 8));
  v35[0] = (__int64)v36;
  sub_2610960(v35, v23[0], (__int64)&v23[0][(unsigned __int64)v23[1]]);
  v37[0] = (__int64)v38;
  sub_2610960(v37, v25, (__int64)v25 + v26);
  v39 = _mm_loadu_si128(&v28);
  sub_B180C0((__int64)&v40, (unsigned __int64)v35);
  if ( (_QWORD *)v37[0] != v38 )
    j_j___libc_free_0(v37[0]);
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0]);
  sub_B18290((__int64)&v40, " will not be inlined into ", 0x1Au);
  v3 = (unsigned __int8 *)sub_B491C0(**(_QWORD **)(a2 + 16));
  sub_B16080((__int64)v29, "Caller", 6, v3);
  v35[0] = (__int64)v36;
  sub_2610960(v35, v29[0], (__int64)&v29[0][(unsigned __int64)v29[1]]);
  v37[0] = (__int64)v38;
  sub_2610960(v37, v31, (__int64)v31 + v32);
  v39 = _mm_loadu_si128(&v34);
  sub_B180C0((__int64)&v40, (unsigned __int64)v35);
  if ( (_QWORD *)v37[0] != v38 )
    j_j___libc_free_0(v37[0]);
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0]);
  sub_B18290((__int64)&v40, " because its definition is unavailable", 0x26u);
  v4 = (__m128i *)(a1 + 96);
  sub_B17B40((__int64)&v40);
  v9 = v41;
  v10 = _mm_loadu_si128(&v44);
  *(_QWORD *)(a1 + 80) = a1 + 96;
  v11 = _mm_loadu_si128(&v46);
  v12 = _mm_loadu_si128(&v47);
  *(_DWORD *)(a1 + 8) = v9;
  LOBYTE(v9) = v42;
  v13 = v49;
  *(__m128i *)(a1 + 24) = v10;
  *(_BYTE *)(a1 + 12) = v9;
  v14 = v43;
  *(__m128i *)(a1 + 48) = v11;
  *(_QWORD *)(a1 + 16) = v14;
  *(__m128i *)(a1 + 64) = v12;
  *(_QWORD *)a1 = &unk_49D9D40;
  *(_QWORD *)(a1 + 40) = v45;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  if ( v13 )
  {
    v20 = v13;
    if ( v13 > 4 )
    {
      sub_11F02D0(a1 + 80, v13, v5, v6, v7, v8);
      v4 = *(__m128i **)(a1 + 80);
      v20 = v49;
    }
    v21 = v48;
    v22 = &v48[10 * v20];
    if ( v48 != v22 )
    {
      do
      {
        if ( v4 )
        {
          v4->m128i_i64[0] = (__int64)v4[1].m128i_i64;
          sub_2610960(v4->m128i_i64, (_BYTE *)*v21, *v21 + v21[1]);
          v4[2].m128i_i64[0] = (__int64)v4[3].m128i_i64;
          sub_2610960(v4[2].m128i_i64, (_BYTE *)v21[4], v21[4] + v21[5]);
          v4[4] = _mm_loadu_si128((const __m128i *)v21 + 4);
        }
        v21 += 10;
        v4 += 5;
      }
      while ( v22 != v21 );
    }
    *(_DWORD *)(a1 + 88) = v13;
  }
  v15 = v31;
  *(_BYTE *)(a1 + 416) = v50[320];
  *(_DWORD *)(a1 + 420) = v51;
  *(_QWORD *)(a1 + 424) = v52;
  *(_QWORD *)a1 = &unk_49D9DB0;
  if ( v15 != &v33 )
    j_j___libc_free_0((unsigned __int64)v15);
  if ( (__int64 *)v29[0] != &v30 )
    j_j___libc_free_0((unsigned __int64)v29[0]);
  if ( v25 != &v27 )
    j_j___libc_free_0((unsigned __int64)v25);
  if ( (__int64 *)v23[0] != &v24 )
    j_j___libc_free_0((unsigned __int64)v23[0]);
  v16 = v48;
  v40 = &unk_49D9D40;
  v17 = &v48[10 * v49];
  if ( v48 != v17 )
  {
    do
    {
      v17 -= 10;
      v18 = v17[4];
      if ( (unsigned __int64 *)v18 != v17 + 6 )
        j_j___libc_free_0(v18);
      if ( (unsigned __int64 *)*v17 != v17 + 2 )
        j_j___libc_free_0(*v17);
    }
    while ( v16 != v17 );
    v17 = v48;
  }
  if ( v17 != (unsigned __int64 *)v50 )
    _libc_free((unsigned __int64)v17);
  return a1;
}
