// Function: sub_1EAE8E0
// Address: 0x1eae8e0
//
void __fastcall sub_1EAE8E0(__int64 **a1, __int64 *a2, unsigned __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  const __m128i *v13; // rbx
  const __m128i *v14; // r15
  const __m128i *v15; // rdi
  __m128i *v16; // rbx
  __m128i *v17; // r12
  __m128i *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rsi
  __m128i *v22; // rbx
  const __m128i *v23; // r15
  int v24; // [rsp+Ch] [rbp-484h]
  const __m128i *v25; // [rsp+10h] [rbp-480h]
  __m128i v26; // [rsp+20h] [rbp-470h] BYREF
  __int64 v27; // [rsp+30h] [rbp-460h]
  _BYTE *v28[2]; // [rsp+40h] [rbp-450h] BYREF
  __int64 v29; // [rsp+50h] [rbp-440h] BYREF
  __int64 *v30; // [rsp+60h] [rbp-430h]
  __int64 v31; // [rsp+68h] [rbp-428h]
  __int64 v32; // [rsp+70h] [rbp-420h] BYREF
  __m128i v33; // [rsp+80h] [rbp-410h] BYREF
  __int64 v34; // [rsp+90h] [rbp-400h]
  __m128i v35; // [rsp+A0h] [rbp-3F0h] BYREF
  __int64 v36; // [rsp+B0h] [rbp-3E0h] BYREF
  __m128i v37; // [rsp+B8h] [rbp-3D8h] BYREF
  __int64 v38; // [rsp+C8h] [rbp-3C8h]
  const char *v39; // [rsp+D0h] [rbp-3C0h] BYREF
  _BYTE v40[24]; // [rsp+D8h] [rbp-3B8h]
  __int64 v41; // [rsp+F0h] [rbp-3A0h]
  __m128i *v42; // [rsp+F8h] [rbp-398h] BYREF
  __int64 v43; // [rsp+100h] [rbp-390h]
  _BYTE v44[356]; // [rsp+108h] [rbp-388h] BYREF
  int v45; // [rsp+26Ch] [rbp-224h]
  __int64 v46; // [rsp+270h] [rbp-220h]
  void *v47; // [rsp+280h] [rbp-210h] BYREF
  __int64 v48; // [rsp+288h] [rbp-208h]
  __int64 v49; // [rsp+290h] [rbp-200h]
  __m128i v50; // [rsp+298h] [rbp-1F8h] BYREF
  __int64 v51; // [rsp+2A8h] [rbp-1E8h]
  const char *v52; // [rsp+2B0h] [rbp-1E0h]
  __m128i v53; // [rsp+2B8h] [rbp-1D8h] BYREF
  __int64 v54; // [rsp+2C8h] [rbp-1C8h]
  char v55; // [rsp+2D0h] [rbp-1C0h]
  const __m128i *v56; // [rsp+2D8h] [rbp-1B8h]
  __int64 v57; // [rsp+2E0h] [rbp-1B0h]
  _BYTE v58[352]; // [rsp+2E8h] [rbp-1A8h] BYREF
  char v59; // [rsp+448h] [rbp-48h]
  int v60; // [rsp+44Ch] [rbp-44h]
  __int64 v61; // [rsp+450h] [rbp-40h]

  v5 = sub_15E0530(**a1);
  if ( sub_1602790(v5)
    || (v19 = sub_15E0530(**a1),
        v20 = sub_16033E0(v19),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 48LL))(v20)) )
  {
    v6 = a2[41];
    v7 = sub_1626D20(*a2);
    sub_15C9150((const char **)&v26, v7);
    v8 = _mm_loadu_si128(&v26);
    v9 = **(_QWORD **)(v6 + 56);
    v48 = 0x200000010LL;
    v10 = *a3;
    v61 = v6;
    v49 = v9;
    v50 = v8;
    v51 = v27;
    v52 = "prologepilog";
    v53.m128i_i64[0] = (__int64)"StackSize";
    v56 = (const __m128i *)v58;
    v57 = 0x400000000LL;
    v53.m128i_i64[1] = 9;
    v55 = 0;
    v47 = &unk_49FD6A0;
    v59 = 0;
    v60 = -1;
    sub_15C9D40((__int64)v28, "NumStackBytes", 13, v10);
    v35.m128i_i64[0] = (__int64)&v36;
    sub_1EAD700(v35.m128i_i64, v28[0], (__int64)&v28[0][(unsigned __int64)v28[1]]);
    v37.m128i_i64[1] = (__int64)&v39;
    sub_1EAD700(&v37.m128i_i64[1], v30, (__int64)v30 + v31);
    *(__m128i *)&v40[8] = _mm_loadu_si128(&v33);
    v41 = v34;
    sub_15CAC60((__int64)&v47, &v35);
    if ( (const char **)v37.m128i_i64[1] != &v39 )
      j_j___libc_free_0(v37.m128i_i64[1], v39 + 1);
    if ( (__int64 *)v35.m128i_i64[0] != &v36 )
      j_j___libc_free_0(v35.m128i_i64[0], v36 + 1);
    sub_15CAB20((__int64)&v47, " stack bytes in function", 0x18u);
    v11 = _mm_loadu_si128(&v50);
    v12 = _mm_loadu_si128(&v53);
    v35.m128i_i32[2] = v48;
    v37 = v11;
    v35.m128i_i8[12] = BYTE4(v48);
    *(__m128i *)v40 = v12;
    v36 = v49;
    v38 = v51;
    v35.m128i_i64[0] = (__int64)&unk_49ECF68;
    v39 = v52;
    LOBYTE(v41) = v55;
    if ( v55 )
      *(_QWORD *)&v40[16] = v54;
    v43 = 0x400000000LL;
    v42 = (__m128i *)v44;
    v24 = v57;
    if ( (_DWORD)v57 )
    {
      v21 = (unsigned int)v57;
      v22 = (__m128i *)v44;
      if ( (unsigned int)v57 > 4 )
      {
        sub_14B3F20((__int64)&v42, (unsigned int)v57);
        v22 = v42;
        v21 = (unsigned int)v57;
      }
      v23 = v56;
      v25 = (const __m128i *)((char *)v56 + 88 * v21);
      if ( v56 != v25 )
      {
        do
        {
          if ( v22 )
          {
            v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
            sub_1EAD700(v22->m128i_i64, v23->m128i_i64[0], v23->m128i_i64[0] + v23->m128i_i64[1]);
            v22[2].m128i_i64[0] = (__int64)v22[3].m128i_i64;
            sub_1EAD700(v22[2].m128i_i64, (_BYTE *)v23[2].m128i_i64[0], v23[2].m128i_i64[0] + v23[2].m128i_i64[1]);
            v22[4] = _mm_loadu_si128(v23 + 4);
            v22[5].m128i_i64[0] = v23[5].m128i_i64[0];
          }
          v23 = (const __m128i *)((char *)v23 + 88);
          v22 = (__m128i *)((char *)v22 + 88);
        }
        while ( v25 != v23 );
      }
      LODWORD(v43) = v24;
    }
    v44[352] = v59;
    v45 = v60;
    v46 = v61;
    v35.m128i_i64[0] = (__int64)&unk_49FD6A0;
    if ( v30 != &v32 )
      j_j___libc_free_0(v30, v32 + 1);
    if ( (__int64 *)v28[0] != &v29 )
      j_j___libc_free_0(v28[0], v29 + 1);
    v13 = v56;
    v47 = &unk_49ECF68;
    v14 = (const __m128i *)((char *)v56 + 88 * (unsigned int)v57);
    if ( v56 != v14 )
    {
      do
      {
        v14 = (const __m128i *)((char *)v14 - 88);
        v15 = (const __m128i *)v14[2].m128i_i64[0];
        if ( v15 != &v14[3] )
          j_j___libc_free_0(v15, v14[3].m128i_i64[0] + 1);
        if ( (const __m128i *)v14->m128i_i64[0] != &v14[1] )
          j_j___libc_free_0(v14->m128i_i64[0], v14[1].m128i_i64[0] + 1);
      }
      while ( v13 != v14 );
      v14 = v56;
    }
    if ( v14 != (const __m128i *)v58 )
      _libc_free((unsigned __int64)v14);
    sub_1E36D90(a1, (__int64)&v35);
    v16 = v42;
    v35.m128i_i64[0] = (__int64)&unk_49ECF68;
    v17 = (__m128i *)((char *)v42 + 88 * (unsigned int)v43);
    if ( v42 != v17 )
    {
      do
      {
        v17 = (__m128i *)((char *)v17 - 88);
        v18 = (__m128i *)v17[2].m128i_i64[0];
        if ( v18 != &v17[3] )
          j_j___libc_free_0(v18, v17[3].m128i_i64[0] + 1);
        if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
          j_j___libc_free_0(v17->m128i_i64[0], v17[1].m128i_i64[0] + 1);
      }
      while ( v16 != v17 );
      v17 = v42;
    }
    if ( v17 != (__m128i *)v44 )
      _libc_free((unsigned __int64)v17);
  }
}
