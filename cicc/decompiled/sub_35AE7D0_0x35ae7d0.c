// Function: sub_35AE7D0
// Address: 0x35ae7d0
//
__int64 __fastcall sub_35AE7D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 *v6; // rax
  char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // eax
  __m128i v14; // xmm3
  __m128i *v15; // r14
  __m128i v16; // xmm4
  unsigned int v17; // ebx
  __m128i v18; // xmm5
  __int64 v19; // rax
  __int64 *v20; // rdi
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r13
  unsigned __int64 v23; // rdi
  __int64 v25; // rsi
  unsigned __int64 *v26; // r15
  unsigned __int64 *v27; // r13
  __m128i v28; // [rsp+20h] [rbp-2E0h] BYREF
  _BYTE *v29[2]; // [rsp+30h] [rbp-2D0h] BYREF
  __int64 v30; // [rsp+40h] [rbp-2C0h] BYREF
  __int64 *v31; // [rsp+50h] [rbp-2B0h]
  __int64 v32; // [rsp+58h] [rbp-2A8h]
  __int64 v33; // [rsp+60h] [rbp-2A0h] BYREF
  __m128i v34; // [rsp+70h] [rbp-290h] BYREF
  _BYTE *v35[2]; // [rsp+80h] [rbp-280h] BYREF
  __int64 v36; // [rsp+90h] [rbp-270h] BYREF
  __int64 *v37; // [rsp+A0h] [rbp-260h]
  __int64 v38; // [rsp+A8h] [rbp-258h]
  __int64 v39; // [rsp+B0h] [rbp-250h] BYREF
  __m128i v40; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v41[2]; // [rsp+D0h] [rbp-230h] BYREF
  _QWORD v42[2]; // [rsp+E0h] [rbp-220h] BYREF
  __int64 v43[2]; // [rsp+F0h] [rbp-210h] BYREF
  _QWORD v44[2]; // [rsp+100h] [rbp-200h] BYREF
  __m128i v45; // [rsp+110h] [rbp-1F0h]
  void *v46; // [rsp+120h] [rbp-1E0h] BYREF
  __int64 v47; // [rsp+128h] [rbp-1D8h]
  __int64 v48; // [rsp+130h] [rbp-1D0h]
  __m128i v49; // [rsp+138h] [rbp-1C8h] BYREF
  const char *v50; // [rsp+148h] [rbp-1B8h]
  __m128i v51; // [rsp+150h] [rbp-1B0h] BYREF
  __m128i v52; // [rsp+160h] [rbp-1A0h] BYREF
  unsigned __int64 *v53; // [rsp+170h] [rbp-190h]
  __int64 v54; // [rsp+178h] [rbp-188h]
  _BYTE v55[320]; // [rsp+180h] [rbp-180h] BYREF
  char v56; // [rsp+2C0h] [rbp-40h]
  int v57; // [rsp+2C4h] [rbp-3Ch]
  __int64 v58; // [rsp+2C8h] [rbp-38h]

  v3 = *(_QWORD *)(*(_QWORD *)a2 + 328LL);
  v4 = sub_B92180(**(_QWORD **)a2);
  sub_B15890(&v28, v4);
  v5 = **(_QWORD **)(v3 + 32);
  v49 = _mm_loadu_si128(&v28);
  v58 = v3;
  v48 = v5;
  v50 = "prologepilog";
  v51.m128i_i64[0] = (__int64)"StackSize";
  v53 = (unsigned __int64 *)v55;
  v54 = 0x400000000LL;
  v47 = 0x200000015LL;
  v52.m128i_i8[8] = 0;
  v46 = &unk_4A28EB8;
  v6 = *(unsigned __int64 **)(a2 + 8);
  v51.m128i_i64[1] = 9;
  v56 = 0;
  v57 = -1;
  sub_B16B10((__int64 *)v29, "NumStackBytes", 13, *v6);
  v41[0] = (__int64)v42;
  sub_35ABC00(v41, v29[0], (__int64)&v29[0][(unsigned __int64)v29[1]]);
  v43[0] = (__int64)v44;
  sub_35ABC00(v43, v31, (__int64)v31 + v32);
  v45 = _mm_loadu_si128(&v34);
  sub_B180C0((__int64)&v46, (unsigned __int64)v41);
  if ( (_QWORD *)v43[0] != v44 )
    j_j___libc_free_0(v43[0]);
  if ( (_QWORD *)v41[0] != v42 )
    j_j___libc_free_0(v41[0]);
  sub_B18290((__int64)&v46, " stack bytes in function '", 0x1Au);
  v7 = (char *)sub_BD5D20(**(_QWORD **)a2);
  sub_B16430((__int64)v35, "Function", 8u, v7, v8);
  v41[0] = (__int64)v42;
  sub_35ABC00(v41, v35[0], (__int64)&v35[0][(unsigned __int64)v35[1]]);
  v43[0] = (__int64)v44;
  sub_35ABC00(v43, v37, (__int64)v37 + v38);
  v45 = _mm_loadu_si128(&v40);
  sub_B180C0((__int64)&v46, (unsigned __int64)v41);
  if ( (_QWORD *)v43[0] != v44 )
    j_j___libc_free_0(v43[0]);
  if ( (_QWORD *)v41[0] != v42 )
    j_j___libc_free_0(v41[0]);
  sub_B18290((__int64)&v46, "'", 1u);
  v13 = v47;
  v14 = _mm_loadu_si128(&v49);
  v15 = (__m128i *)(a1 + 96);
  v16 = _mm_loadu_si128(&v51);
  v17 = v54;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_DWORD *)(a1 + 8) = v13;
  LOBYTE(v13) = BYTE4(v47);
  v18 = _mm_loadu_si128(&v52);
  *(__m128i *)(a1 + 24) = v14;
  *(_BYTE *)(a1 + 12) = v13;
  v19 = v48;
  *(__m128i *)(a1 + 48) = v16;
  *(_QWORD *)(a1 + 16) = v19;
  *(__m128i *)(a1 + 64) = v18;
  *(_QWORD *)a1 = &unk_49D9D40;
  *(_QWORD *)(a1 + 40) = v50;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  if ( v17 )
  {
    v25 = v17;
    if ( v17 > 4 )
    {
      sub_11F02D0(a1 + 80, v17, v9, v10, v11, v12);
      v15 = *(__m128i **)(a1 + 80);
      v25 = (unsigned int)v54;
    }
    v26 = v53;
    v27 = &v53[10 * v25];
    if ( v53 != v27 )
    {
      do
      {
        if ( v15 )
        {
          v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
          sub_35ABC00(v15->m128i_i64, (_BYTE *)*v26, *v26 + v26[1]);
          v15[2].m128i_i64[0] = (__int64)v15[3].m128i_i64;
          sub_35ABC00(v15[2].m128i_i64, (_BYTE *)v26[4], v26[4] + v26[5]);
          v15[4] = _mm_loadu_si128((const __m128i *)v26 + 4);
        }
        v26 += 10;
        v15 += 5;
      }
      while ( v27 != v26 );
    }
    *(_DWORD *)(a1 + 88) = v17;
  }
  v20 = v37;
  *(_BYTE *)(a1 + 416) = v56;
  *(_DWORD *)(a1 + 420) = v57;
  *(_QWORD *)(a1 + 424) = v58;
  *(_QWORD *)a1 = &unk_4A28EB8;
  if ( v20 != &v39 )
    j_j___libc_free_0((unsigned __int64)v20);
  if ( (__int64 *)v35[0] != &v36 )
    j_j___libc_free_0((unsigned __int64)v35[0]);
  if ( v31 != &v33 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( (__int64 *)v29[0] != &v30 )
    j_j___libc_free_0((unsigned __int64)v29[0]);
  v21 = v53;
  v46 = &unk_49D9D40;
  v22 = &v53[10 * (unsigned int)v54];
  if ( v53 != v22 )
  {
    do
    {
      v22 -= 10;
      v23 = v22[4];
      if ( (unsigned __int64 *)v23 != v22 + 6 )
        j_j___libc_free_0(v23);
      if ( (unsigned __int64 *)*v22 != v22 + 2 )
        j_j___libc_free_0(*v22);
    }
    while ( v21 != v22 );
    v22 = v53;
  }
  if ( v22 != (unsigned __int64 *)v55 )
    _libc_free((unsigned __int64)v22);
  return a1;
}
