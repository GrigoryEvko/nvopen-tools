// Function: sub_1B1E040
// Address: 0x1b1e040
//
void __fastcall sub_1B1E040(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rdx
  __int8 v14; // cl
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r15
  unsigned int v20; // r14d
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // r8d
  int v24; // r9d
  const __m128i *v25; // r14
  __int32 v26; // eax
  unsigned __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // r8d
  int v30; // r9d
  __int64 v31; // rax
  _QWORD *v32; // rbx
  _QWORD *v33; // r13
  unsigned __int64 v34; // rdi
  __int64 v35; // rax
  unsigned __int64 v36; // r15
  __int64 *v37; // rax
  __int64 v38; // rsi
  unsigned __int64 v39; // rsi
  __m128i *p_dest; // rdi
  unsigned __int64 v41; // [rsp+8h] [rbp-118h]
  __m128i v42; // [rsp+10h] [rbp-110h] BYREF
  __m128i dest; // [rsp+20h] [rbp-100h] BYREF
  __int64 i; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v45[2]; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE v46[128]; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v47; // [rsp+C8h] [rbp-58h]
  _QWORD *v48; // [rsp+D0h] [rbp-50h]
  __int64 v49; // [rsp+D8h] [rbp-48h]
  unsigned int v50; // [rsp+E0h] [rbp-40h]

  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 40) = 128;
  v11 = (_QWORD *)sub_22077B0(0x2000);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = v11;
  v42.m128i_i64[1] = 2;
  v12 = *(unsigned int *)(a1 + 40);
  dest.m128i_i64[0] = 0;
  dest.m128i_i64[1] = -8;
  v13 = &v11[8 * v12];
  v42.m128i_i64[0] = (__int64)&unk_49E6B50;
  for ( i = 0; v13 != v11; v11 += 8 )
  {
    if ( v11 )
    {
      v14 = v42.m128i_i8[8];
      v11[2] = 0;
      v11[3] = -8;
      *v11 = &unk_49E6B50;
      v11[1] = v14 & 6;
      v11[4] = i;
    }
  }
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x400000000LL;
  *(_BYTE *)(a1 + 89) = 1;
  sub_14585E0(a1 + 176);
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 488) = a2;
  *(_QWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_DWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_DWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 496) = a4;
  *(_QWORD *)(a1 + 504) = a5;
  *(_QWORD *)(a1 + 512) = a6;
  if ( a7 )
  {
    v42.m128i_i64[1] = 0x400000000LL;
    v42.m128i_i64[0] = (__int64)&dest;
    v19 = a2[1];
    v20 = *(_DWORD *)(v19 + 280);
    if ( v20 && &v42 != (__m128i *)(v19 + 272) )
    {
      p_dest = &dest;
      v15 = 16LL * v20;
      if ( v20 <= 4
        || (sub_16CD150((__int64)&v42, &dest, v20, 16, v20, v18),
            p_dest = (__m128i *)v42.m128i_i64[0],
            (v15 = 16LL * *(unsigned int *)(v19 + 280)) != 0) )
      {
        memcpy(p_dest, *(const void **)(v19 + 272), v15);
      }
      v42.m128i_i32[2] = v20;
    }
    sub_1B1DC30(a1, (__int64)&v42, v15, v16, v17, v18);
    if ( (__m128i *)v42.m128i_i64[0] != &dest )
      _libc_free(v42.m128i_u64[0]);
    v25 = (const __m128i *)sub_1458800(*a2);
    v42.m128i_i64[1] = v25->m128i_i64[1];
    dest = _mm_loadu_si128(v25 + 1);
    v26 = v25[2].m128i_i32[0];
    v45[0] = (unsigned __int64)v46;
    LODWORD(i) = v26;
    v42.m128i_i64[0] = (__int64)&unk_49EC708;
    v45[1] = 0x1000000000LL;
    if ( v25[3].m128i_i32[0] )
      sub_1B1DA60((__int64)v45, (__int64)&v25[2].m128i_i64[1], v21, v22, v23, v24);
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    j___libc_free_0(0);
    v31 = v25[13].m128i_u32[0];
    v50 = v31;
    if ( (_DWORD)v31 )
    {
      v35 = sub_22077B0(56 * v31);
      v48 = (_QWORD *)v35;
      LODWORD(v49) = v25[12].m128i_i32[2];
      v27 = v25[12].m128i_u32[3];
      HIDWORD(v49) = v25[12].m128i_i32[3];
      if ( v50 )
      {
        v29 = 0;
        v36 = 0;
        v27 = 0;
        while ( 1 )
        {
          v37 = (__int64 *)(v36 + v35);
          if ( v37 )
          {
            *v37 = *(_QWORD *)(v25[12].m128i_i64[0] + v36);
            v37 = &v48[v36 / 8];
          }
          v28 = *v37;
          if ( *v37 != -16 && v28 != -8 )
          {
            v38 = v25[12].m128i_i64[0];
            v37[2] = 0x400000000LL;
            v37[1] = (__int64)(v37 + 3);
            v39 = v36 + v38;
            v28 = *(unsigned int *)(v39 + 16);
            if ( (_DWORD)v28 )
            {
              v41 = v27;
              sub_1B1DA60((__int64)(v37 + 1), v39 + 8, v27, v28, 0, v30);
              v27 = v41;
              v29 = 0;
            }
          }
          ++v27;
          v36 += 56LL;
          if ( v50 <= v27 )
            break;
          v35 = (__int64)v48;
        }
      }
    }
    else
    {
      v48 = 0;
      v49 = 0;
    }
    sub_1B1DDA0(a1, &v42, v27, v28, v29, v30);
    v42.m128i_i64[0] = (__int64)&unk_49EC708;
    if ( v50 )
    {
      v32 = v48;
      v33 = &v48[7 * v50];
      do
      {
        if ( *v32 != -16 && *v32 != -8 )
        {
          v34 = v32[1];
          if ( (_QWORD *)v34 != v32 + 3 )
            _libc_free(v34);
        }
        v32 += 7;
      }
      while ( v33 != v32 );
    }
    j___libc_free_0(v48);
    if ( (_BYTE *)v45[0] != v46 )
      _libc_free(v45[0]);
  }
}
