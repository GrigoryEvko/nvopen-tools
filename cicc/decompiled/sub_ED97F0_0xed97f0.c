// Function: sub_ED97F0
// Address: 0xed97f0
//
_DWORD *__fastcall sub_ED97F0(__int64 a1, int a2, _DWORD *a3, char a4)
{
  unsigned int v6; // r13d
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  unsigned int v9; // edx
  __int64 v10; // rax
  const __m128i *v11; // rsi
  bool v12; // zf
  __m128i *v13; // rax
  unsigned int v14; // r15d
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 *v17; // rdx
  __int64 v18; // rcx
  __int64 *v19; // rax
  int v20; // r15d
  __int64 v21; // rax
  __int64 v22; // r14
  const __m128i *v23; // rcx
  const __m128i *v24; // rdi
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __m128i *v27; // rsi
  __m128i *v28; // rdx
  const __m128i *v29; // rax
  __int64 v30; // r15
  __int64 v31; // rdi
  _DWORD *v32; // r13
  __int64 v33; // r13
  char *v34; // rax
  char *v35; // r14
  _BYTE *v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rdi
  __int64 v40; // r12
  __int64 v41; // rdi
  unsigned __int64 v43; // [rsp+0h] [rbp-110h]
  __int64 v44; // [rsp+8h] [rbp-108h]
  __int64 v45; // [rsp+10h] [rbp-100h]
  __int64 v46; // [rsp+18h] [rbp-F8h]
  __int64 v47; // [rsp+20h] [rbp-F0h]
  __int64 v48; // [rsp+28h] [rbp-E8h]
  __int64 v49; // [rsp+30h] [rbp-E0h]
  __int64 *v50; // [rsp+38h] [rbp-D8h]
  __int64 v51; // [rsp+48h] [rbp-C8h] BYREF
  const __m128i *v52; // [rsp+50h] [rbp-C0h] BYREF
  __m128i *v53; // [rsp+58h] [rbp-B8h] BYREF
  const __m128i *v54; // [rsp+60h] [rbp-B0h]
  __m128i **v55; // [rsp+68h] [rbp-A8h]
  __m128i **v56; // [rsp+70h] [rbp-A0h]
  __int64 v57; // [rsp+78h] [rbp-98h]
  _BYTE *v58; // [rsp+80h] [rbp-90h]
  char *v59; // [rsp+88h] [rbp-88h]
  char *v60; // [rsp+90h] [rbp-80h]
  __int64 v61; // [rsp+98h] [rbp-78h]
  __int64 v62; // [rsp+A0h] [rbp-70h]
  __int64 v63; // [rsp+A8h] [rbp-68h]
  __int64 v64; // [rsp+B0h] [rbp-60h]
  __int64 v65; // [rsp+B8h] [rbp-58h]
  __int64 v66; // [rsp+C0h] [rbp-50h]
  __int64 v67; // [rsp+C8h] [rbp-48h]
  __int64 v68; // [rsp+D0h] [rbp-40h]

  if ( a2 <= 3 )
  {
    v33 = 4LL * unk_497B318;
    if ( (unsigned __int64)(4LL * unk_497B318) > 0x7FFFFFFFFFFFFFFCLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    if ( v33 )
    {
      v34 = (char *)sub_22077B0(4LL * unk_497B318);
      v35 = &v34[v33];
      v36 = memcpy(v34, &unk_3F88320, 4LL * unk_497B318);
    }
    else
    {
      v35 = 0;
      v36 = 0;
    }
    v60 = v35;
    v55 = &v53;
    v56 = &v53;
    LODWORD(v53) = 0;
    v54 = 0;
    v57 = 0;
    v58 = v36;
    v59 = v35;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    sub_EFA340(&v51, &v52);
    v37 = v51;
    v51 = 0;
    v38 = *(_QWORD *)(a1 + 144);
    *(_QWORD *)(a1 + 144) = v37;
    if ( v38 )
    {
      v39 = *(_QWORD *)(v38 + 8);
      if ( v39 )
        j_j___libc_free_0(v39, *(_QWORD *)(v38 + 24) - v39);
      j_j___libc_free_0(v38, 88);
      v40 = v51;
      if ( v51 )
      {
        v41 = *(_QWORD *)(v51 + 8);
        if ( v41 )
          j_j___libc_free_0(v41, *(_QWORD *)(v51 + 24) - v41);
        j_j___libc_free_0(v40, 88);
      }
    }
    if ( v61 )
      j_j___libc_free_0(v61, v63 - v61);
    if ( v58 )
      j_j___libc_free_0(v58, v60 - v58);
    v32 = a3;
    sub_ED7670((__int64)v54);
  }
  else
  {
    v6 = 8 * (3 * a3[2] + *a3 + 2);
    v7 = (_QWORD *)sub_22077B0(v6);
    v8 = v7;
    if ( v7 )
      memset(v7, 0, v6);
    v9 = 0;
    v10 = 0;
    v11 = (const __m128i *)((unsigned __int64)v6 >> 3);
    if ( v11 )
    {
      do
      {
        v8[v10] = *(_QWORD *)&a3[2 * v10];
        v10 = ++v9;
      }
      while ( v9 < (unsigned __int64)v11 );
    }
    v12 = v8[1] == 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    if ( !v12 )
    {
      v11 = 0;
      v13 = 0;
      v14 = 0;
      v15 = 0;
      while ( 1 )
      {
        v17 = &v8[3 * v15 + 2 + *v8];
        v18 = *v17;
        LODWORD(v51) = *v17;
        if ( v11 == v13 )
        {
          sub_ED9630(&v52, v11, &v51, v17 + 1, v17 + 2);
          v15 = ++v14;
          if ( (unsigned __int64)v14 >= v8[1] )
            break;
        }
        else
        {
          if ( v13 )
          {
            v11 = (const __m128i *)v17[1];
            v16 = v17[2];
            v13->m128i_i32[0] = v18;
            v13->m128i_i64[1] = (__int64)v11;
            v13[1].m128i_i64[0] = v16;
            v13 = v53;
          }
          v53 = (__m128i *)((char *)v13 + 24);
          v15 = ++v14;
          if ( (unsigned __int64)v14 >= v8[1] )
            break;
        }
        v13 = v53;
        v11 = v54;
      }
    }
    v48 = v8[2];
    v45 = v8[3];
    v47 = v8[4];
    v46 = v8[6];
    v49 = v8[5];
    v44 = v8[7];
    if ( a4 )
    {
      v19 = (__int64 *)(a1 + 152);
      v20 = 1;
    }
    else
    {
      v19 = (__int64 *)(a1 + 144);
      v20 = 0;
    }
    v50 = v19;
    v21 = sub_22077B0(88);
    v22 = v21;
    if ( v21 )
    {
      v23 = v53;
      v24 = v52;
      *(_DWORD *)v21 = v20;
      *(_QWORD *)(v21 + 8) = 0;
      *(_QWORD *)(v21 + 16) = 0;
      *(_QWORD *)(v21 + 24) = 0;
      v25 = (char *)v23 - (char *)v24;
      if ( v23 == v24 )
      {
        v27 = 0;
      }
      else
      {
        if ( v25 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(v24, v11, v25);
        v43 = (char *)v23 - (char *)v24;
        v26 = sub_22077B0((char *)v23 - (char *)v24);
        v23 = v53;
        v24 = v52;
        v25 = v43;
        v27 = (__m128i *)v26;
      }
      *(_QWORD *)(v22 + 8) = v27;
      *(_QWORD *)(v22 + 16) = v27;
      *(_QWORD *)(v22 + 24) = (char *)v27 + v25;
      if ( v24 != v23 )
      {
        v28 = v27;
        v29 = v24;
        do
        {
          if ( v28 )
          {
            *v28 = _mm_loadu_si128(v29);
            v28[1].m128i_i64[0] = v29[1].m128i_i64[0];
          }
          v29 = (const __m128i *)((char *)v29 + 24);
          v28 = (__m128i *)((char *)v28 + 24);
        }
        while ( v29 != v23 );
        v27 = (__m128i *)((char *)v27 + 8 * ((unsigned __int64)((char *)&v29[-2].m128i_u64[1] - (char *)v24) >> 3) + 24);
      }
      *(_QWORD *)(v22 + 16) = v27;
      *(_BYTE *)(v22 + 72) = 0;
      *(_QWORD *)(v22 + 32) = v44;
      *(_QWORD *)(v22 + 80) = 0;
      *(_QWORD *)(v22 + 40) = v49;
      *(_QWORD *)(v22 + 48) = v46;
      *(_QWORD *)(v22 + 56) = v47;
      *(_DWORD *)(v22 + 64) = v45;
      *(_DWORD *)(v22 + 68) = v48;
    }
    v30 = *v50;
    *v50 = v22;
    if ( v30 )
    {
      v31 = *(_QWORD *)(v30 + 8);
      if ( v31 )
        j_j___libc_free_0(v31, *(_QWORD *)(v30 + 24) - v31);
      j_j___libc_free_0(v30, 88);
    }
    v32 = &a3[v6 / 4];
    if ( v52 )
      j_j___libc_free_0(v52, (char *)v54 - (char *)v52);
    j___libc_free_0(v8);
  }
  return v32;
}
