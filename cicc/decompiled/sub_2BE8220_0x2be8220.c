// Function: sub_2BE8220
// Address: 0x2be8220
//
void __fastcall sub_2BE8220(__int64 a1, char a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rax
  const void *v4; // r9
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // r14
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // rax
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  int v14; // ecx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  _BYTE *v20; // rax
  _BYTE *v21; // rsi
  unsigned __int64 *v22; // [rsp+8h] [rbp-218h]
  const void *v23; // [rsp+10h] [rbp-210h]
  __int64 v24; // [rsp+18h] [rbp-208h]
  __int64 v25; // [rsp+20h] [rbp-200h]
  __int64 v26; // [rsp+28h] [rbp-1F8h]
  __int64 v27; // [rsp+30h] [rbp-1F0h]
  char v28; // [rsp+38h] [rbp-1E8h]
  __int64 v29; // [rsp+40h] [rbp-1E0h]
  __int64 v30; // [rsp+48h] [rbp-1D8h]
  _BYTE *v31; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v32; // [rsp+58h] [rbp-1C8h]
  __int16 v33; // [rsp+6Eh] [rbp-1B2h] BYREF
  __m128i v34; // [rsp+70h] [rbp-1B0h] BYREF
  unsigned __int64 v35; // [rsp+80h] [rbp-1A0h]
  __m128i v36; // [rsp+90h] [rbp-190h] BYREF
  __int64 (__fastcall *v37)(unsigned __int64 **, unsigned __int64 **, int); // [rsp+A0h] [rbp-180h]
  bool (__fastcall *v38)(_QWORD *, _BYTE *); // [rsp+A8h] [rbp-178h]
  unsigned __int64 v39; // [rsp+B0h] [rbp-170h] BYREF
  _BYTE *v40; // [rsp+B8h] [rbp-168h]
  _BYTE *v41; // [rsp+C0h] [rbp-160h]
  unsigned __int64 *v42; // [rsp+C8h] [rbp-158h]
  unsigned __int64 *v43; // [rsp+D0h] [rbp-150h]
  __int64 v44; // [rsp+D8h] [rbp-148h]
  unsigned __int64 v45; // [rsp+E0h] [rbp-140h]
  __int64 v46; // [rsp+E8h] [rbp-138h]
  __int64 v47; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v48; // [rsp+F8h] [rbp-128h]
  __int64 v49; // [rsp+100h] [rbp-120h]
  __int64 v50; // [rsp+108h] [rbp-118h]
  __int64 v51; // [rsp+110h] [rbp-110h]
  __int64 v52; // [rsp+118h] [rbp-108h]
  char v53; // [rsp+120h] [rbp-100h]
  __m128i v54; // [rsp+128h] [rbp-F8h] BYREF
  __m128i v55; // [rsp+138h] [rbp-E8h] BYREF
  char v56[96]; // [rsp+150h] [rbp-D0h] BYREF
  int v57; // [rsp+1B0h] [rbp-70h]
  __m128i v58; // [rsp+1C8h] [rbp-58h] BYREF
  __m128i v59[4]; // [rsp+1D8h] [rbp-48h] BYREF

  v2 = *(_QWORD *)(a1 + 384);
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = v2;
  v53 = a2;
  v54 = 0u;
  v55 = 0u;
  v33 = 0;
  if ( (*(_BYTE *)a1 & 0x10) == 0 )
  {
    if ( (unsigned __int8)sub_2BE0770(a1) )
    {
      v20 = *(_BYTE **)(a1 + 272);
      LOBYTE(v33) = 1;
      HIBYTE(v33) = *v20;
    }
    else if ( *(_DWORD *)(a1 + 152) == 28 && (unsigned __int8)sub_2BE0030(a1) )
    {
      v33 = 11521;
    }
  }
  while ( (unsigned __int8)sub_2BE7B90(a1, (unsigned __int8 *)&v33, (__int64)&v39) )
    ;
  if ( (_BYTE)v33 )
  {
    v21 = v40;
    v56[0] = HIBYTE(v33);
    if ( v40 == v41 )
    {
      sub_17EB120((__int64)&v39, v40, v56);
    }
    else
    {
      if ( v40 )
      {
        *v40 = HIBYTE(v33);
        v21 = v40;
      }
      v40 = v21 + 1;
    }
  }
  sub_2BE3840((__int64)&v39);
  v3 = v40;
  v4 = v41;
  v40 = 0;
  v41 = 0;
  v31 = v3;
  v23 = v4;
  v30 = v46;
  v22 = *(unsigned __int64 **)(a1 + 256);
  v24 = v44;
  v5 = v39;
  v25 = v47;
  v6 = (unsigned __int64)v42;
  v26 = v49;
  v7 = v43;
  v27 = v50;
  v8 = v45;
  v32 = v48;
  v39 = 0;
  v44 = 0;
  v43 = 0;
  v42 = 0;
  v47 = 0;
  v46 = 0;
  v45 = 0;
  v50 = 0;
  v49 = 0;
  v9 = _mm_loadu_si128(&v54);
  v48 = 0;
  v57 = v51;
  v10 = _mm_loadu_si128(&v55);
  v28 = v53;
  v37 = 0;
  v29 = v52;
  v58 = v9;
  v59[0] = v10;
  v11 = sub_22077B0(0x98u);
  if ( v11 )
  {
    v12 = _mm_loadu_si128(&v58);
    *(_QWORD *)(v11 + 40) = v24;
    v13 = _mm_loadu_si128(v59);
    *(_QWORD *)(v11 + 16) = v23;
    *(_QWORD *)(v11 + 8) = v31;
    *(_QWORD *)(v11 + 56) = v30;
    *(_QWORD *)(v11 + 64) = v25;
    *(_QWORD *)(v11 + 72) = v32;
    v14 = v57;
    *(_QWORD *)(v11 + 80) = v26;
    *(_DWORD *)(v11 + 96) = v14;
    *(_QWORD *)(v11 + 88) = v27;
    *(_QWORD *)(v11 + 104) = v29;
    *(_BYTE *)(v11 + 112) = v28;
    v32 = 0;
    *(_QWORD *)v11 = v5;
    v5 = 0;
    *(_QWORD *)(v11 + 24) = v6;
    v6 = 0;
    *(_QWORD *)(v11 + 32) = v7;
    v7 = 0;
    *(_QWORD *)(v11 + 48) = v8;
    v8 = 0;
    *(__m128i *)(v11 + 120) = v12;
    *(__m128i *)(v11 + 136) = v13;
  }
  v36.m128i_i64[0] = v11;
  v38 = sub_2BDB730;
  v37 = sub_2BDD2E0;
  v15 = sub_2BE0EB0(v22, &v36);
  v16 = *(_QWORD *)(a1 + 256);
  v34.m128i_i64[1] = v15;
  v35 = v15;
  v34.m128i_i64[0] = v16;
  sub_2BE3490((unsigned __int64 *)(a1 + 304), &v34);
  if ( v37 )
    v37((unsigned __int64 **)&v36, (unsigned __int64 **)&v36, 3);
  if ( v32 )
    j_j___libc_free_0(v32);
  if ( v8 )
    j_j___libc_free_0(v8);
  for ( i = (unsigned __int64 *)v6; v7 != i; i += 4 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  if ( v5 )
    j_j___libc_free_0(v5);
  if ( v48 )
    j_j___libc_free_0(v48);
  if ( v45 )
    j_j___libc_free_0(v45);
  v18 = v43;
  v19 = v42;
  if ( v43 != v42 )
  {
    do
    {
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        j_j___libc_free_0(*v19);
      v19 += 4;
    }
    while ( v18 != v19 );
    v19 = v42;
  }
  if ( v19 )
    j_j___libc_free_0((unsigned __int64)v19);
  if ( v39 )
    j_j___libc_free_0(v39);
}
