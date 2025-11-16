// Function: sub_37DEEC0
// Address: 0x37deec0
//
__int64 __fastcall sub_37DEEC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned int v10; // esi
  __int64 v11; // rdi
  _DWORD *v12; // r14
  __int64 v13; // r13
  __int64 v14; // rcx
  __int16 v15; // ax
  __int64 v16; // r8
  const __m128i *v17; // rbx
  const __m128i *v18; // r14
  __m128i v19; // xmm1
  __m128i v20; // xmm0
  __int64 v21; // rax
  char v22; // al
  __m128i *v23; // r10
  __int32 v24; // r13d
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // r10
  __int64 v29; // rsi
  unsigned int v30; // eax
  __int64 v31; // rsi
  bool v32; // al
  __int64 v33; // rdi
  unsigned int v35; // esi
  int v36; // eax
  int v37; // eax
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  __m128i *v41; // r11
  __m128i *v42; // rax
  char v43; // al
  unsigned __int64 v44; // r8
  __int64 v45; // rdi
  const void *v46; // rsi
  unsigned int *v47; // [rsp+0h] [rbp-130h]
  __int64 v48; // [rsp+0h] [rbp-130h]
  __int64 v49; // [rsp+10h] [rbp-120h]
  int v50; // [rsp+18h] [rbp-118h]
  __m128i *v51; // [rsp+18h] [rbp-118h]
  __m128i *v52; // [rsp+18h] [rbp-118h]
  __m128i *v53; // [rsp+20h] [rbp-110h] BYREF
  __m128i *v54; // [rsp+28h] [rbp-108h] BYREF
  _QWORD v55[6]; // [rsp+30h] [rbp-100h] BYREF
  __m128i v56; // [rsp+60h] [rbp-D0h]
  __m128i v57; // [rsp+70h] [rbp-C0h]
  __m128i v58; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v59; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v60; // [rsp+A0h] [rbp-90h] BYREF
  __m128i v61; // [rsp+B0h] [rbp-80h]
  _BYTE *v62; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v63; // [rsp+C8h] [rbp-68h]
  _BYTE v64[96]; // [rsp+D0h] [rbp-60h] BYREF

  v8 = *(_QWORD *)(a2 + 32);
  v9 = v8 + 40;
  if ( *(_WORD *)(a2 + 68) != 14 )
  {
    v9 = v8 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
    v8 += 80;
  }
  for ( ; v9 != v8; *v12 = sub_37BA230(v11, v10) )
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)v8 )
      {
        v10 = *(_DWORD *)(v8 + 8);
        if ( v10 )
        {
          v11 = *(_QWORD *)(a1 + 408);
          v12 = (_DWORD *)(*(_QWORD *)(v11 + 64) + 4LL * v10);
          if ( *v12 == -1 )
            break;
        }
      }
      v8 += 40;
      if ( v9 == v8 )
        goto LABEL_10;
    }
    v8 += 40;
  }
LABEL_10:
  v13 = *(_QWORD *)(a1 + 424);
  if ( v13 )
  {
    v14 = *(_QWORD *)(a2 + 32);
    v62 = v64;
    v63 = 0xC00000000LL;
    v15 = *(_WORD *)(a2 + 68);
    if ( v15 == 14 )
    {
      v17 = (const __m128i *)(v14 + 40);
      v16 = v14;
      goto LABEL_46;
    }
    v16 = v14 + 80;
    v17 = (const __m128i *)(v14 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
    if ( v15 != 15 )
    {
LABEL_13:
      if ( v17 == (const __m128i *)v16 )
        goto LABEL_27;
      v18 = (const __m128i *)v16;
      v49 = a1 + 2232;
      while ( !v18->m128i_i8[0] )
      {
        v27 = *(_QWORD *)(a1 + 408);
        v28 = a1 + 2168;
        v29 = v18->m128i_u32[2];
        v30 = *(_DWORD *)(*(_QWORD *)(v27 + 64) + 4 * v29);
        if ( v30 == -1 )
        {
          v47 = (unsigned int *)(*(_QWORD *)(v27 + 64) + 4 * v29);
          v30 = sub_37BA230(*(_QWORD *)(a1 + 408), v29);
          v28 = a1 + 2168;
          *v47 = v30;
        }
        v31 = *(_QWORD *)(*(_QWORD *)(v27 + 32) + 8LL * v30);
        v56.m128i_i64[0] = v31;
        if ( v31 == unk_5051170 )
        {
          v24 = dword_5051178[0];
LABEL_18:
          v25 = (unsigned int)v63;
          v26 = (unsigned int)v63 + 1LL;
          if ( v26 > HIDWORD(v63) )
            goto LABEL_25;
          goto LABEL_19;
        }
        v59.m128i_i64[0] = v31;
        v24 = sub_37C5950(v28, v31);
        v25 = (unsigned int)v63;
        v26 = (unsigned int)v63 + 1LL;
        if ( v26 > HIDWORD(v63) )
        {
LABEL_25:
          sub_C8D5F0((__int64)&v62, v64, v26, 4u, v16, a6);
          v25 = (unsigned int)v63;
        }
LABEL_19:
        v18 = (const __m128i *)((char *)v18 + 40);
        *(_DWORD *)&v62[4 * v25] = v24;
        LODWORD(v63) = v63 + 1;
        if ( v17 == v18 )
        {
          v13 = *(_QWORD *)(a1 + 424);
          v15 = *(_WORD *)(a2 + 68);
          goto LABEL_27;
        }
      }
      if ( (unsigned __int8)(v18->m128i_i8[0] - 1) > 2u )
        BUG();
      v19 = _mm_loadu_si128(v18);
      v58.m128i_i8[8] = 1;
      v59 = v19;
      v20 = _mm_loadu_si128(v18 + 1);
      v56 = v19;
      v60 = v20;
      v21 = v18[2].m128i_i64[0];
      v57 = v20;
      v58.m128i_i64[0] = v21;
      v50 = *(_DWORD *)(a1 + 2192);
      v61 = _mm_loadu_si128(&v58);
      v22 = sub_37BD360(v49, v59.m128i_i32, &v53);
      v23 = v53;
      if ( v22 )
      {
LABEL_17:
        v24 = v23[2].m128i_i32[2];
        goto LABEL_18;
      }
      v35 = *(_DWORD *)(a1 + 2256);
      v36 = *(_DWORD *)(a1 + 2248);
      v54 = v53;
      ++*(_QWORD *)(a1 + 2232);
      v37 = v36 + 1;
      v16 = 2 * v35;
      if ( 4 * v37 >= 3 * v35 )
      {
        v35 *= 2;
      }
      else if ( v35 - *(_DWORD *)(a1 + 2252) - v37 > v35 >> 3 )
      {
        goto LABEL_36;
      }
      sub_37C5160(v49, v35);
      sub_37BD360(v49, v59.m128i_i32, &v54);
      v23 = v54;
      v37 = *(_DWORD *)(a1 + 2248) + 1;
LABEL_36:
      *(_DWORD *)(a1 + 2248) = v37;
      v55[0] = 21;
      v55[2] = 0;
      if ( (unsigned __int8)(v23->m128i_i8[0] - 21) > 1u )
      {
        v43 = sub_2EAB6C0((__int64)v23, (char *)v55);
        v23 = v54;
        if ( v43 )
          goto LABEL_39;
      }
      else if ( v23->m128i_i8[0] == 21 )
      {
LABEL_39:
        *v23 = _mm_loadu_si128(&v59);
        v23[1] = _mm_loadu_si128(&v60);
        v23[2].m128i_i64[0] = v61.m128i_i64[0];
        v23[2].m128i_i32[2] = 2 * v50 + 1;
        v38 = *(unsigned int *)(a1 + 2192);
        v39 = v38 + 1;
        if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 2196) )
        {
          v44 = *(_QWORD *)(a1 + 2184);
          v45 = a1 + 2184;
          v46 = (const void *)(a1 + 2200);
          if ( v44 > (unsigned __int64)&v59 || (v48 = *(_QWORD *)(a1 + 2184), (unsigned __int64)&v59 >= v44 + 40 * v38) )
          {
            v52 = v23;
            sub_C8D5F0(v45, v46, v39, 0x28u, v44, a6);
            v40 = *(_QWORD *)(a1 + 2184);
            v38 = *(unsigned int *)(a1 + 2192);
            v41 = &v59;
            v23 = v52;
          }
          else
          {
            v51 = v23;
            sub_C8D5F0(v45, v46, v39, 0x28u, v44, a6);
            v16 = v48;
            v40 = *(_QWORD *)(a1 + 2184);
            v38 = *(unsigned int *)(a1 + 2192);
            v23 = v51;
            v41 = (__m128i *)((char *)&v59 + v40 - v48);
          }
        }
        else
        {
          v40 = *(_QWORD *)(a1 + 2184);
          v41 = &v59;
        }
        v42 = (__m128i *)(v40 + 40 * v38);
        *v42 = _mm_loadu_si128(v41);
        v42[1] = _mm_loadu_si128(v41 + 1);
        v42[2].m128i_i64[0] = v41[2].m128i_i64[0];
        ++*(_DWORD *)(a1 + 2192);
        goto LABEL_17;
      }
      --*(_DWORD *)(a1 + 2252);
      goto LABEL_39;
    }
    if ( (const __m128i *)v16 != v17 )
    {
      do
      {
LABEL_46:
        if ( !*(_BYTE *)v16 && !*(_DWORD *)(v16 + 8) )
          goto LABEL_27;
        v16 += 40;
      }
      while ( v17 != (const __m128i *)v16 );
      if ( v15 == 14 )
      {
        v17 = (const __m128i *)(v14 + 40);
        v16 = v14;
      }
      else
      {
        v16 = v14 + 80;
        v17 = (const __m128i *)(v14 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
      }
      goto LABEL_13;
    }
LABEL_27:
    v59.m128i_i8[9] = v15 == 15;
    v59.m128i_i64[0] = sub_2E891C0(a2);
    v32 = 0;
    if ( *(_WORD *)(a2 + 68) == 14 )
      v32 = *(_BYTE *)(*(_QWORD *)(a2 + 32) + 40LL) == 1;
    v59.m128i_i8[8] = v32;
    sub_37DE9A0(v13, a2, &v59, (__int64)&v62);
    if ( v62 != v64 )
      _libc_free((unsigned __int64)v62);
  }
  v33 = *(_QWORD *)(a1 + 432);
  if ( v33 )
    sub_37CE0D0(v33, a2);
  return 1;
}
