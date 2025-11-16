// Function: sub_31822F0
// Address: 0x31822f0
//
_QWORD *__fastcall sub_31822F0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  int v10; // esi
  __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 *v13; // rdx
  __int64 v14; // rsi
  _QWORD *v15; // r12
  unsigned __int16 v16; // ax
  unsigned __int64 *v17; // rbx
  unsigned __int64 *v18; // r13
  unsigned __int64 v19; // rdi
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // r8
  __int64 v27; // rax
  _QWORD **v28; // rdi
  __int64 v29; // rax
  int v30; // ecx
  unsigned __int64 v31; // rdx
  unsigned int v32; // edi
  __int64 v33; // rsi
  unsigned __int64 *v34; // r8
  unsigned __int64 *v35; // r9
  __int64 v36; // rax
  __m128i *v37; // rbx
  _QWORD *v38; // rax
  int v39; // eax
  int v40; // r10d
  int v41; // [rsp+4h] [rbp-FCh]
  char v42; // [rsp+8h] [rbp-F8h]
  char v43; // [rsp+10h] [rbp-F0h]
  __int64 v44; // [rsp+18h] [rbp-E8h]
  __int64 v45; // [rsp+20h] [rbp-E0h]
  __m128i *v49; // [rsp+60h] [rbp-A0h]
  __int64 v50; // [rsp+68h] [rbp-98h]
  __m128i v51; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int64 *v52; // [rsp+80h] [rbp-80h]
  __int64 v53; // [rsp+88h] [rbp-78h]
  _BYTE v54[112]; // [rsp+90h] [rbp-70h] BYREF

  v52 = (unsigned __int64 *)v54;
  v10 = *(_DWORD *)(a6 + 16);
  v53 = 0x100000000LL;
  if ( !v10 )
  {
    v42 = a8;
    v11 = 0;
    v12 = 0;
    LOBYTE(v41) = 0;
    v13 = (unsigned __int64 *)v54;
    v43 = BYTE1(a8);
    goto LABEL_3;
  }
  if ( !a7 )
    BUG();
  v21 = *(unsigned int *)(a6 + 24);
  v22 = *(_QWORD *)(a7 + 16);
  v23 = *(_QWORD *)(a6 + 8);
  if ( !(_DWORD)v21 )
    goto LABEL_34;
  v24 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( v22 != *v25 )
  {
    v39 = 1;
    while ( v26 != -4096 )
    {
      v40 = v39 + 1;
      v24 = (v21 - 1) & (v39 + v24);
      v25 = (__int64 *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( v22 == *v25 )
        goto LABEL_18;
      v39 = v40;
    }
LABEL_34:
    v25 = (__int64 *)(v23 + 16 * v21);
  }
LABEL_18:
  v27 = v25[1];
  v28 = (_QWORD **)(v27 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v27 & 4) != 0 )
    v28 = (_QWORD **)**v28;
  v29 = sub_AA4FF0((__int64)v28);
  if ( !v29 )
    BUG();
  v30 = v53;
  v31 = (unsigned int)*(unsigned __int8 *)(v29 - 24) - 39;
  v32 = v53;
  if ( (unsigned int)v31 <= 0x38 )
  {
    v33 = 0x100060000000001LL;
    if ( _bittest64(&v33, v31) )
    {
      v51.m128i_i64[0] = 0x74656C636E7566LL;
      v49 = &v51;
      v37 = (__m128i *)&v52[7 * (unsigned int)v53];
      v50 = 7;
      if ( v37 )
      {
        v37->m128i_i64[0] = (__int64)v37[1].m128i_i64;
        if ( v49 == &v51 )
        {
          v37[1] = _mm_load_si128(&v51);
        }
        else
        {
          v37->m128i_i64[0] = (__int64)v49;
          v37[1].m128i_i64[0] = v51.m128i_i64[0];
        }
        v44 = v29 - 24;
        v37->m128i_i64[1] = v50;
        v49 = &v51;
        v50 = 0;
        v51.m128i_i8[0] = 0;
        v37[2].m128i_i64[0] = 0;
        v37[2].m128i_i64[1] = 0;
        v37[3].m128i_i64[0] = 0;
        v38 = (_QWORD *)sub_22077B0(8u);
        v37[2].m128i_i64[0] = (__int64)v38;
        v37[3].m128i_i64[0] = (__int64)(v38 + 1);
        *v38 = v44;
        v37[2].m128i_i64[1] = (__int64)(v38 + 1);
        if ( v49 != &v51 )
          j_j___libc_free_0((unsigned __int64)v49);
        v30 = v53;
      }
      v32 = v30 + 1;
      LODWORD(v53) = v30 + 1;
    }
  }
  v13 = v52;
  v12 = v32;
  v11 = 16 * v32;
  v42 = a8;
  LOBYTE(v41) = (_DWORD)v11 != 0;
  v34 = &v52[7 * v12];
  v43 = BYTE1(a8);
  if ( v52 == v34 )
  {
    v10 = 0;
  }
  else
  {
    v35 = v52;
    v10 = 0;
    do
    {
      v36 = v35[5] - v35[4];
      v35 += 7;
      v10 += v36 >> 3;
    }
    while ( v34 != v35 );
  }
LABEL_3:
  v14 = (unsigned int)(v10 + a4 + 1);
  v45 = (__int64)v13;
  v15 = sub_BD2CC0(88, (v11 << 32) | v14);
  if ( v15 )
  {
    LOBYTE(v16) = v42;
    HIBYTE(v16) = v43;
    sub_B44260((__int64)v15, **(_QWORD **)(a1 + 16), 56, v14 & 0x7FFFFFF | (v41 << 28), a7, v16);
    v15[9] = 0;
    sub_B4A290((__int64)v15, a1, a2, a3, a4, a5, v45, v12);
  }
  v17 = v52;
  v18 = &v52[7 * (unsigned int)v53];
  if ( v52 != v18 )
  {
    do
    {
      v19 = *(v18 - 3);
      v18 -= 7;
      if ( v19 )
        j_j___libc_free_0(v19);
      if ( (unsigned __int64 *)*v18 != v18 + 2 )
        j_j___libc_free_0(*v18);
    }
    while ( v17 != v18 );
    v18 = v52;
  }
  if ( v18 != (unsigned __int64 *)v54 )
    _libc_free((unsigned __int64)v18);
  return v15;
}
