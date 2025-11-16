// Function: sub_1F0CE40
// Address: 0x1f0ce40
//
void __fastcall sub_1F0CE40(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        const __m128i *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rdi
  _BYTE *v17; // rsi
  size_t v18; // rdx
  __int64 v19; // rsi
  int v20; // ebx
  const __m128i *v21; // r12
  __m128i *v22; // rbx
  __m128i *v23; // r12
  __m128i *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i *v27; // r15
  const __m128i *v28; // r14
  const __m128i *v29; // r15
  const __m128i *v30; // rdi
  void *v31; // [rsp+10h] [rbp-3F0h] BYREF
  int v32; // [rsp+18h] [rbp-3E8h]
  char v33; // [rsp+1Ch] [rbp-3E4h]
  __int64 v34; // [rsp+20h] [rbp-3E0h]
  __m128i v35; // [rsp+28h] [rbp-3D8h]
  __int64 v36; // [rsp+38h] [rbp-3C8h]
  char *v37; // [rsp+40h] [rbp-3C0h]
  __m128i v38; // [rsp+48h] [rbp-3B8h]
  __int64 v39; // [rsp+58h] [rbp-3A8h]
  char v40; // [rsp+60h] [rbp-3A0h]
  __m128i *v41; // [rsp+68h] [rbp-398h] BYREF
  __int64 v42; // [rsp+70h] [rbp-390h]
  _BYTE v43[352]; // [rsp+78h] [rbp-388h] BYREF
  char v44; // [rsp+1D8h] [rbp-228h]
  int v45; // [rsp+1DCh] [rbp-224h]
  __int64 v46; // [rsp+1E0h] [rbp-220h]
  void *v47; // [rsp+1F0h] [rbp-210h] BYREF
  __int64 v48; // [rsp+1F8h] [rbp-208h]
  __int64 v49; // [rsp+200h] [rbp-200h]
  __m128i v50; // [rsp+208h] [rbp-1F8h] BYREF
  __int64 v51; // [rsp+218h] [rbp-1E8h]
  char *v52; // [rsp+220h] [rbp-1E0h]
  __m128i v53; // [rsp+228h] [rbp-1D8h] BYREF
  __int64 v54; // [rsp+238h] [rbp-1C8h]
  char v55; // [rsp+240h] [rbp-1C0h]
  const __m128i *v56; // [rsp+248h] [rbp-1B8h]
  __int64 v57; // [rsp+250h] [rbp-1B0h]
  _BYTE v58[352]; // [rsp+258h] [rbp-1A8h] BYREF
  char v59; // [rsp+3B8h] [rbp-48h]
  int v60; // [rsp+3BCh] [rbp-44h]
  __int64 v61; // [rsp+3C0h] [rbp-40h]

  v11 = sub_15E0530(**a1);
  if ( !sub_1602790(v11) )
  {
    v25 = sub_15E0530(**a1);
    v26 = sub_16033E0(v25);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v26 + 48LL))(v26) )
      return;
  }
  v12 = _mm_loadu_si128(a8);
  v13 = *a7;
  v14 = a8[1].m128i_i64[0];
  v15 = a7[1];
  v16 = **(_QWORD **)(*a9 + 56);
  v61 = *a9;
  v51 = v14;
  v53.m128i_i64[0] = v13;
  v53.m128i_i64[1] = v15;
  v47 = &unk_49FC050;
  v48 = 0x20000000FLL;
  v17 = *(_BYTE **)a10;
  v18 = *(_QWORD *)(a10 + 8);
  v49 = v16;
  v56 = (const __m128i *)v58;
  v52 = "shrink-wrap";
  v55 = 0;
  v57 = 0x400000000LL;
  v59 = 0;
  v60 = -1;
  v50 = v12;
  sub_15CAB20((__int64)&v47, v17, v18);
  v32 = v48;
  v35 = _mm_loadu_si128(&v50);
  v33 = BYTE4(v48);
  v38 = _mm_loadu_si128(&v53);
  v34 = v49;
  v36 = v51;
  v31 = &unk_49ECF68;
  v37 = v52;
  v40 = v55;
  if ( v55 )
    v39 = v54;
  v19 = (unsigned int)v57;
  v41 = (__m128i *)v43;
  v20 = v57;
  v42 = 0x400000000LL;
  if ( (_DWORD)v57 )
  {
    if ( (unsigned int)v57 > 4uLL )
    {
      sub_14B3F20((__int64)&v41, (unsigned int)v57);
      v27 = v41;
      v19 = (unsigned int)v57;
    }
    else
    {
      v27 = (__m128i *)v43;
    }
    v28 = v56;
    v21 = (const __m128i *)((char *)v56 + 88 * v19);
    if ( v56 != v21 )
    {
      do
      {
        if ( v27 )
        {
          v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
          sub_1F0BFD0(v27->m128i_i64, v28->m128i_i64[0], v28->m128i_i64[0] + v28->m128i_i64[1]);
          v27[2].m128i_i64[0] = (__int64)v27[3].m128i_i64;
          sub_1F0BFD0(v27[2].m128i_i64, (_BYTE *)v28[2].m128i_i64[0], v28[2].m128i_i64[0] + v28[2].m128i_i64[1]);
          v27[4] = _mm_loadu_si128(v28 + 4);
          v27[5].m128i_i64[0] = v28[5].m128i_i64[0];
        }
        v28 = (const __m128i *)((char *)v28 + 88);
        v27 = (__m128i *)((char *)v27 + 88);
      }
      while ( v21 != v28 );
      v21 = v56;
      LODWORD(v42) = v20;
      v29 = (const __m128i *)((char *)v56 + 88 * (unsigned int)v57);
      v44 = v59;
      v45 = v60;
      v46 = v61;
      v31 = &unk_49FC050;
      v47 = &unk_49ECF68;
      if ( v56 != v29 )
      {
        do
        {
          v29 = (const __m128i *)((char *)v29 - 88);
          v30 = (const __m128i *)v29[2].m128i_i64[0];
          if ( v30 != &v29[3] )
            j_j___libc_free_0(v30, v29[3].m128i_i64[0] + 1);
          if ( (const __m128i *)v29->m128i_i64[0] != &v29[1] )
            j_j___libc_free_0(v29->m128i_i64[0], v29[1].m128i_i64[0] + 1);
        }
        while ( v21 != v29 );
        v21 = v56;
      }
      goto LABEL_7;
    }
    LODWORD(v42) = v20;
  }
  else
  {
    v21 = v56;
  }
  v44 = v59;
  v45 = v60;
  v46 = v61;
  v31 = &unk_49FC050;
LABEL_7:
  if ( v21 != (const __m128i *)v58 )
    _libc_free((unsigned __int64)v21);
  sub_1E36D90(a1, (__int64)&v31);
  v22 = v41;
  v31 = &unk_49ECF68;
  v23 = (__m128i *)((char *)v41 + 88 * (unsigned int)v42);
  if ( v41 != v23 )
  {
    do
    {
      v23 = (__m128i *)((char *)v23 - 88);
      v24 = (__m128i *)v23[2].m128i_i64[0];
      if ( v24 != &v23[3] )
        j_j___libc_free_0(v24, v23[3].m128i_i64[0] + 1);
      if ( (__m128i *)v23->m128i_i64[0] != &v23[1] )
        j_j___libc_free_0(v23->m128i_i64[0], v23[1].m128i_i64[0] + 1);
    }
    while ( v22 != v23 );
    v23 = v41;
  }
  if ( v23 != (__m128i *)v43 )
    _libc_free((unsigned __int64)v23);
}
