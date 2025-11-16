// Function: sub_B40C20
// Address: 0xb40c20
//
__int64 __fastcall sub_B40C20(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  char v7; // dl
  char v8; // al
  char v9; // al
  __int64 v10; // r13
  unsigned int v12; // eax
  int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // edi
  __m128i *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // r14
  __m128i *v21; // r12
  __int64 v22; // rdx
  __int16 v23; // ax
  __int64 v24; // rax
  size_t v25; // rdx
  __int64 v26; // rax
  bool v27; // zf
  __m128i *v28; // rax
  __m128i *v29; // rcx
  __int64 v30; // rdx
  __int16 v31; // ax
  __int64 v32; // rdi
  int v33; // r13d
  int v34; // eax
  char src; // [rsp+8h] [rbp-A8h]
  __m128i *srca; // [rsp+8h] [rbp-A8h]
  __int64 v38; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+18h] [rbp-98h] BYREF
  __m128i *v40; // [rsp+20h] [rbp-90h] BYREF
  size_t v41; // [rsp+28h] [rbp-88h]
  __m128i v42; // [rsp+30h] [rbp-80h] BYREF
  char v43; // [rsp+40h] [rbp-70h]
  char v44; // [rsp+41h] [rbp-6Fh]
  int v45; // [rsp+48h] [rbp-68h]
  __m128i *v46; // [rsp+50h] [rbp-60h] BYREF
  size_t n; // [rsp+58h] [rbp-58h]
  __m128i v48; // [rsp+60h] [rbp-50h] BYREF
  char v49; // [rsp+70h] [rbp-40h]
  char v50; // [rsp+71h] [rbp-3Fh]
  int v51; // [rsp+78h] [rbp-38h]

  v5 = *(_BYTE **)a2;
  v6 = *(_QWORD *)(a2 + 8);
  v40 = &v42;
  sub_B3AE60((__int64 *)&v40, v5, (__int64)&v5[v6]);
  v7 = *(_BYTE *)(a2 + 32);
  v8 = *(_BYTE *)(a2 + 33);
  v45 = 0;
  v43 = v7;
  v44 = v8;
  v46 = &v48;
  if ( v40 == &v42 )
  {
    v48 = _mm_load_si128(&v42);
  }
  else
  {
    v46 = v40;
    v48.m128i_i64[0] = v42.m128i_i64[0];
  }
  v49 = v7;
  v42.m128i_i8[0] = 0;
  n = v41;
  v40 = &v42;
  v41 = 0;
  v50 = v8;
  v51 = 0;
  v9 = sub_B3C4F0(a1, (__int64)&v46, &v38);
  v10 = v38;
  if ( v9 )
  {
    src = 0;
    goto LABEL_5;
  }
  v12 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v39 = v10;
  v13 = (v12 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v15 = 96;
    v14 = 32;
  }
  else
  {
    v14 = *(_DWORD *)(a1 + 24);
    v15 = 3 * v14;
  }
  if ( 4 * v13 >= v15 )
  {
    v14 *= 2;
LABEL_46:
    sub_B3F770(a1, v14);
    sub_B3C4F0(a1, (__int64)&v46, &v39);
    v10 = v39;
    v13 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
    goto LABEL_15;
  }
  if ( v14 - (v13 + *(_DWORD *)(a1 + 12)) <= v14 >> 3 )
    goto LABEL_46;
LABEL_15:
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v13);
  if ( *(_WORD *)(v10 + 32) || *(_BYTE *)(v10 + 32) && !*(_BYTE *)(v10 + 33) && *(_QWORD *)(v10 + 8) )
    --*(_DWORD *)(a1 + 12);
  v16 = *(__m128i **)v10;
  if ( v46 == &v48 )
  {
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
        v16->m128i_i8[0] = v48.m128i_i8[0];
      else
        memcpy(v16, &v48, n);
      v25 = n;
      v16 = *(__m128i **)v10;
    }
    *(_QWORD *)(v10 + 8) = v25;
    v16->m128i_i8[v25] = 0;
    v16 = v46;
    goto LABEL_21;
  }
  if ( v16 == (__m128i *)(v10 + 16) )
  {
    *(_QWORD *)v10 = v46;
    *(_QWORD *)(v10 + 8) = n;
    *(_QWORD *)(v10 + 16) = v48.m128i_i64[0];
    goto LABEL_33;
  }
  *(_QWORD *)v10 = v46;
  v17 = *(_QWORD *)(v10 + 16);
  *(_QWORD *)(v10 + 8) = n;
  *(_QWORD *)(v10 + 16) = v48.m128i_i64[0];
  if ( !v16 )
  {
LABEL_33:
    v46 = &v48;
    v16 = &v48;
    goto LABEL_21;
  }
  v46 = v16;
  v48.m128i_i64[0] = v17;
LABEL_21:
  n = 0;
  v16->m128i_i8[0] = 0;
  src = 1;
  *(_BYTE *)(v10 + 32) = v49;
  *(_BYTE *)(v10 + 33) = v50;
  *(_DWORD *)(v10 + 40) = v51;
LABEL_5:
  if ( v46 != &v48 )
    j_j___libc_free_0(v46, v48.m128i_i64[0] + 1);
  if ( v40 != &v42 )
    j_j___libc_free_0(v40, v42.m128i_i64[0] + 1);
  if ( !src )
    return *(_QWORD *)(a1 + 1552) + 832LL * *(unsigned int *)(v10 + 40);
  *(_DWORD *)(v10 + 40) = *(_DWORD *)(a1 + 1560);
  v18 = *(unsigned int *)(a1 + 1560);
  v19 = v18;
  if ( *(_DWORD *)(a1 + 1564) <= (unsigned int)v18 )
  {
    v20 = sub_C8D7D0(a1 + 1552, a1 + 1568, 0, 832, &v46);
    v26 = 832LL * *(unsigned int *)(a1 + 1560);
    v27 = v20 + v26 == 0;
    v28 = (__m128i *)(v20 + v26);
    v29 = v28;
    if ( !v27 )
    {
      v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
      if ( *(_QWORD *)a2 == a2 + 16 )
      {
        v28[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
      }
      else
      {
        v28->m128i_i64[0] = *(_QWORD *)a2;
        v28[1].m128i_i64[0] = *(_QWORD *)(a2 + 16);
      }
      v30 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)a2 = a2 + 16;
      *(_QWORD *)(a2 + 8) = 0;
      v28->m128i_i64[1] = v30;
      v31 = *(_WORD *)(a2 + 32);
      *(_BYTE *)(a2 + 16) = 0;
      v29[2].m128i_i16[0] = v31;
      v29[2].m128i_i64[1] = (__int64)&v29[3].m128i_i64[1];
      v29[3].m128i_i64[0] = 0x400000000LL;
      if ( *(_DWORD *)(a3 + 8) )
      {
        srca = v29;
        sub_B3E030(&v29[2].m128i_i64[1], a3);
        v29 = srca;
      }
      v29[51].m128i_i64[1] = *(_QWORD *)(a3 + 784);
    }
    sub_B3F040((const __m128i **)(a1 + 1552), v20);
    v32 = *(_QWORD *)(a1 + 1552);
    v33 = (int)v46;
    if ( a1 + 1568 != v32 )
      _libc_free(v32, v20);
    v34 = *(_DWORD *)(a1 + 1560);
    *(_QWORD *)(a1 + 1552) = v20;
    *(_DWORD *)(a1 + 1564) = v33;
    v24 = (unsigned int)(v34 + 1);
    *(_DWORD *)(a1 + 1560) = v24;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 1552);
    v21 = (__m128i *)(v20 + 832 * v18);
    if ( v21 )
    {
      v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
      if ( *(_QWORD *)a2 == a2 + 16 )
      {
        v21[1] = _mm_loadu_si128((const __m128i *)(a2 + 16));
      }
      else
      {
        v21->m128i_i64[0] = *(_QWORD *)a2;
        v21[1].m128i_i64[0] = *(_QWORD *)(a2 + 16);
      }
      v22 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)a2 = a2 + 16;
      *(_QWORD *)(a2 + 8) = 0;
      v21->m128i_i64[1] = v22;
      v23 = *(_WORD *)(a2 + 32);
      *(_BYTE *)(a2 + 16) = 0;
      v21[2].m128i_i16[0] = v23;
      v21[2].m128i_i64[1] = (__int64)&v21[3].m128i_i64[1];
      v21[3].m128i_i64[0] = 0x400000000LL;
      if ( *(_DWORD *)(a3 + 8) )
        sub_B3E030(&v21[2].m128i_i64[1], a3);
      v21[51].m128i_i64[1] = *(_QWORD *)(a3 + 784);
      v19 = *(_DWORD *)(a1 + 1560);
      v20 = *(_QWORD *)(a1 + 1552);
    }
    v24 = (unsigned int)(v19 + 1);
    *(_DWORD *)(a1 + 1560) = v24;
  }
  return v20 + 832 * v24 - 832;
}
