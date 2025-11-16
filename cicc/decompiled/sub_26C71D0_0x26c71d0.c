// Function: sub_26C71D0
// Address: 0x26c71d0
//
void __fastcall sub_26C71D0(__m128i *a1, char a2)
{
  const __m128i *v2; // rbx
  size_t v3; // rdx
  int *v4; // rsi
  size_t v5; // r15
  unsigned __int64 v6; // rcx
  __m128i **v7; // rax
  __m128i *v8; // r12
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // r14
  _QWORD *v11; // rdi
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // r13
  _QWORD *v16; // rdi
  _QWORD *v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  unsigned __int64 v22; // r15
  __m128i v23; // xmm1
  _QWORD *v24; // rdx
  __m128i v25; // xmm2
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  int v29; // ecx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  int v32; // ecx
  unsigned __int64 v33; // r9
  __m128i **v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // [rsp+0h] [rbp-230h]
  __int64 v39; // [rsp+40h] [rbp-1F0h]
  char v40; // [rsp+50h] [rbp-1E0h]
  unsigned __int64 v41; // [rsp+50h] [rbp-1E0h]
  unsigned __int64 v42; // [rsp+60h] [rbp-1D0h] BYREF
  _QWORD *v43; // [rsp+70h] [rbp-1C0h] BYREF
  unsigned __int64 v44; // [rsp+78h] [rbp-1B8h]
  __int64 v45; // [rsp+80h] [rbp-1B0h]
  __int64 v46; // [rsp+88h] [rbp-1A8h]
  __m128i v47; // [rsp+90h] [rbp-1A0h] BYREF
  _QWORD v48[2]; // [rsp+A0h] [rbp-190h] BYREF
  _QWORD v49[48]; // [rsp+B0h] [rbp-180h] BYREF

  v2 = (const __m128i *)a1[1].m128i_i64[0];
  v43 = v48;
  v44 = 1;
  v45 = 0;
  v46 = 0;
  v47.m128i_i32[0] = 1065353216;
  v47.m128i_i64[1] = 0;
  v48[0] = 0;
  if ( a2 )
  {
    if ( v2 )
    {
      while ( 1 )
      {
        v3 = v2[2].m128i_u64[1];
        v4 = (int *)v2[2].m128i_i64[0];
        v5 = v3;
        memset(&v49[20], 0, 0xB0u);
        v49[34] = 0;
        v6 = v3;
        v39 = v3;
        v49[32] = &v49[30];
        v49[33] = &v49[30];
        v49[37] = 0;
        v49[38] = &v49[36];
        v49[39] = &v49[36];
        v49[40] = 0;
        if ( v4 )
        {
          sub_C7D030(v49);
          sub_C7D280((int *)v49, v4, v5);
          sub_C7D290(v49, &v42);
          v6 = v42;
        }
        v49[0] = v6;
        v7 = (__m128i **)sub_C1DD00(&v43, v6 % v44, v49, v6);
        if ( v7 )
        {
          v8 = *v7;
          if ( *v7 )
            break;
        }
        v21 = (_QWORD *)sub_22077B0(0xC8u);
        v22 = (unsigned __int64)v21;
        if ( v21 )
          *v21 = 0;
        v23 = _mm_loadu_si128((const __m128i *)&v49[22]);
        v24 = v21 + 12;
        v25 = _mm_loadu_si128((const __m128i *)&v49[24]);
        v21[1] = v49[0];
        v26 = v49[20];
        *(__m128i *)(v22 + 32) = v23;
        *(_QWORD *)(v22 + 16) = v26;
        v27 = v49[21];
        *(__m128i *)(v22 + 48) = v25;
        *(_QWORD *)(v22 + 24) = v27;
        *(_QWORD *)(v22 + 64) = v49[26];
        *(_QWORD *)(v22 + 72) = v49[27];
        *(_QWORD *)(v22 + 80) = v49[28];
        v28 = v49[31];
        if ( v49[31] )
        {
          v29 = v49[30];
          *(_QWORD *)(v22 + 104) = v49[31];
          *(_DWORD *)(v22 + 96) = v29;
          *(_QWORD *)(v22 + 112) = v49[32];
          *(_QWORD *)(v22 + 120) = v49[33];
          *(_QWORD *)(v28 + 8) = v24;
          v49[31] = 0;
          *(_QWORD *)(v22 + 128) = v49[34];
          v49[34] = 0;
          v49[32] = &v49[30];
          v49[33] = &v49[30];
        }
        else
        {
          *(_DWORD *)(v22 + 96) = 0;
          *(_QWORD *)(v22 + 104) = 0;
          *(_QWORD *)(v22 + 112) = v24;
          *(_QWORD *)(v22 + 120) = v24;
          *(_QWORD *)(v22 + 128) = 0;
        }
        v30 = v49[37];
        v31 = v22 + 144;
        if ( v49[37] )
        {
          v32 = v49[36];
          *(_QWORD *)(v22 + 152) = v49[37];
          *(_DWORD *)(v22 + 144) = v32;
          *(_QWORD *)(v22 + 160) = v49[38];
          *(_QWORD *)(v22 + 168) = v49[39];
          *(_QWORD *)(v30 + 8) = v31;
          v49[37] = 0;
          *(_QWORD *)(v22 + 176) = v49[40];
          v49[40] = 0;
          v49[38] = &v49[36];
          v49[39] = &v49[36];
        }
        else
        {
          *(_DWORD *)(v22 + 144) = 0;
          *(_QWORD *)(v22 + 152) = 0;
          *(_QWORD *)(v22 + 160) = v31;
          *(_QWORD *)(v22 + 168) = v31;
          *(_QWORD *)(v22 + 176) = 0;
        }
        v33 = *(_QWORD *)(v22 + 8);
        *(_QWORD *)(v22 + 184) = v49[41];
        v36 = v33;
        v41 = v33 % v44;
        v34 = (__m128i **)sub_C1DD00(&v43, v33 % v44, (_QWORD *)(v22 + 8), v33);
        if ( v34 && (v8 = *v34) != 0 )
        {
          sub_26BC990(*(_QWORD **)(v22 + 152));
          sub_26BB480(*(_QWORD **)(v22 + 104));
          j_j___libc_free_0(v22);
        }
        else
        {
          v8 = (__m128i *)sub_26BB140((unsigned __int64 *)&v43, v41, v36, (_QWORD *)v22, 1);
        }
        v9 = v49[37];
        v40 = a2;
        if ( v49[37] )
          goto LABEL_8;
        sub_26BB480((_QWORD *)v49[31]);
LABEL_10:
        v8[3].m128i_i64[0] = 0;
        v8[3].m128i_i64[1] = 0;
        v8[2].m128i_i64[0] = (__int64)v4;
        v8[4].m128i_i64[0] = 0;
        v8[2].m128i_i64[1] = v39;
LABEL_11:
        sub_C1D5C0(v8 + 1, v2 + 1, 1u);
        v2 = (const __m128i *)v2->m128i_i64[0];
        if ( !v2 )
          goto LABEL_12;
      }
      v9 = v49[37];
      v40 = 0;
      if ( !v49[37] )
      {
        sub_26BB480((_QWORD *)v49[31]);
        goto LABEL_11;
      }
      do
      {
LABEL_8:
        v10 = v9;
        sub_26BC990(*(_QWORD **)(v9 + 24));
        v11 = *(_QWORD **)(v9 + 56);
        v9 = *(_QWORD *)(v9 + 16);
        sub_26BCBE0(v11);
        j_j___libc_free_0(v10);
      }
      while ( v9 );
      sub_26BB480((_QWORD *)v49[31]);
      if ( !v40 )
        goto LABEL_11;
      goto LABEL_10;
    }
  }
  else if ( v2 )
  {
    do
    {
      sub_EFA6B0(&v43, (__int64)v2[1].m128i_i64);
      v2 = (const __m128i *)v2->m128i_i64[0];
    }
    while ( v2 );
LABEL_12:
    v12 = (_QWORD *)a1[1].m128i_i64[0];
    while ( v12 )
    {
      v13 = (unsigned __int64)v12;
      v12 = (_QWORD *)*v12;
      v14 = *(_QWORD *)(v13 + 152);
      while ( v14 )
      {
        v15 = v14;
        sub_26BC990(*(_QWORD **)(v14 + 24));
        v16 = *(_QWORD **)(v14 + 56);
        v14 = *(_QWORD *)(v14 + 16);
        sub_26BCBE0(v16);
        j_j___libc_free_0(v15);
      }
      sub_26BB480(*(_QWORD **)(v13 + 104));
      j_j___libc_free_0(v13);
    }
  }
  if ( (__m128i *)a1->m128i_i64[0] != &a1[3] )
    j_j___libc_free_0(a1->m128i_i64[0]);
  v17 = v43;
  a1[2] = _mm_loadu_si128(&v47);
  if ( v17 == v48 )
  {
    v35 = v48[0];
    a1->m128i_i64[0] = (__int64)a1[3].m128i_i64;
    a1[3].m128i_i64[0] = v35;
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v17;
  }
  v18 = v44;
  v19 = v45;
  v20 = v46;
  a1->m128i_i64[1] = v44;
  a1[1].m128i_i64[0] = v19;
  a1[1].m128i_i64[1] = v20;
  if ( v19 )
    *(_QWORD *)(a1->m128i_i64[0] + 8 * (*(_QWORD *)(v19 + 192) % v18)) = a1 + 1;
  v47.m128i_i64[1] = 0;
  v44 = 1;
  v48[0] = 0;
  v43 = v48;
  v45 = 0;
  v46 = 0;
  sub_26C2AF0((__int64)&v43);
  if ( v43 != v48 )
    j_j___libc_free_0((unsigned __int64)v43);
}
