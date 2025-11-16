// Function: sub_34AADC0
// Address: 0x34aadc0
//
void __fastcall sub_34AADC0(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r15
  __int32 v7; // eax
  __int64 v8; // rdx
  unsigned int *v9; // r10
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // eax
  unsigned int *v13; // r15
  unsigned int v14; // esi
  unsigned __int64 v15; // r8
  __int64 v16; // rdx
  unsigned __int64 *v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // r8
  __m128i *v25; // r13
  unsigned int v26; // eax
  int v27; // eax
  unsigned int v28; // esi
  __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  const __m128i *v35; // [rsp+8h] [rbp-F8h]
  __int64 v37; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v38; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v39; // [rsp+20h] [rbp-E0h]
  unsigned int *v40; // [rsp+28h] [rbp-D8h]
  __m128i *v41; // [rsp+30h] [rbp-D0h] BYREF
  __m128i *v42; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v43[3]; // [rsp+40h] [rbp-C0h] BYREF
  char v44; // [rsp+58h] [rbp-A8h]
  __int64 v45; // [rsp+60h] [rbp-A0h]
  __m128i v46; // [rsp+70h] [rbp-90h] BYREF
  __m128i v47; // [rsp+80h] [rbp-80h] BYREF
  __int64 v48; // [rsp+90h] [rbp-70h]
  char *v49; // [rsp+98h] [rbp-68h] BYREF
  __int64 v50; // [rsp+A0h] [rbp-60h]
  _BYTE v51[88]; // [rsp+A8h] [rbp-58h] BYREF

  v6 = a3;
  v7 = a3[3].m128i_i32[2];
  v8 = a1 + 816;
  v9 = *(unsigned int **)a2;
  if ( (unsigned int)(v7 - 2) > 1 )
    v8 = a1 + 224;
  v37 = v8;
  v10 = *(unsigned int *)(a2 + 8);
  v11 = (__int64)&v9[2 * v10];
  v12 = *(_DWORD *)(a2 + 8);
  v40 = (unsigned int *)v11;
  if ( (unsigned int *)v11 != v9 )
  {
    v35 = v6;
    v13 = *(unsigned int **)a2;
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 208);
      v15 = v13[1] | ((unsigned __int64)*v13 << 32);
      if ( v14 )
        break;
      v16 = *(unsigned int *)(a1 + 212);
      if ( (_DWORD)v16 == 11 )
      {
        v46.m128i_i64[0] = a1 + 16;
        v47.m128i_i64[0] = 0x400000000LL;
        v23 = (unsigned __int64 *)(a1 + 24);
        v46.m128i_i64[1] = (__int64)&v47.m128i_i64[1];
        do
        {
          if ( v15 <= *v23 )
            break;
          ++v14;
          v23 += 2;
        }
        while ( v14 != 11 );
        v38 = v15;
        sub_34A26E0((__int64)&v46, v14, v16, v11, v15, a6);
        v24 = v38;
LABEL_24:
        sub_34A8E00((__int64)&v46, v24, v24, 0);
        if ( (unsigned __int64 *)v46.m128i_i64[1] != &v47.m128i_u64[1] )
          _libc_free(v46.m128i_u64[1]);
        goto LABEL_12;
      }
      if ( (_DWORD)v16 )
      {
        v17 = (unsigned __int64 *)(a1 + 24);
        do
        {
          if ( v15 <= *v17 )
            break;
          ++v14;
          v17 += 2;
        }
        while ( (_DWORD)v16 != v14 );
      }
      v46.m128i_i32[0] = v14;
      *(_DWORD *)(a1 + 212) = sub_34A32D0(a1 + 16, (unsigned int *)&v46, v16, v15, v15, 0);
LABEL_12:
      v13 += 2;
      if ( v40 == v13 )
      {
        v6 = v35;
        v12 = *(_DWORD *)(a2 + 8);
        goto LABEL_14;
      }
    }
    v39 = v13[1] | ((unsigned __int64)*v13 << 32);
    v46.m128i_i64[0] = a1 + 16;
    v46.m128i_i64[1] = (__int64)&v47.m128i_i64[1];
    v47.m128i_i64[0] = 0x400000000LL;
    sub_34A3C90((__int64)&v46, v15, v10, v11, v15, a6);
    v24 = v39;
    goto LABEL_24;
  }
LABEL_14:
  v18 = _mm_loadu_si128(v6);
  v19 = _mm_loadu_si128(v6 + 1);
  v20 = v6[2].m128i_i64[0];
  v49 = v51;
  v50 = 0x200000000LL;
  v48 = v20;
  v46 = v18;
  v47 = v19;
  if ( v12 )
    sub_349DD80((__int64)&v49, a2, v20, 0x200000000LL, a5, a6);
  if ( !(unsigned __int8)sub_34A1150(v37, (__int64)&v46, (__int64 *)&v41) )
  {
    v25 = v41;
    v26 = *(_DWORD *)(v37 + 8);
    ++*(_QWORD *)v37;
    v42 = v25;
    v27 = (v26 >> 1) + 1;
    if ( (*(_BYTE *)(v37 + 8) & 1) != 0 )
    {
      v29 = 24;
      v28 = 8;
    }
    else
    {
      v28 = *(_DWORD *)(v37 + 24);
      v29 = 3 * v28;
    }
    v30 = (unsigned int)(4 * v27);
    if ( (unsigned int)v30 >= (unsigned int)v29 )
    {
      v28 *= 2;
    }
    else
    {
      v30 = v28 - (v27 + *(_DWORD *)(v37 + 12));
      v29 = v28 >> 3;
      if ( (unsigned int)v30 > (unsigned int)v29 )
      {
LABEL_31:
        v43[0] = 0;
        v44 = 0;
        v45 = 0;
        *(_DWORD *)(v37 + 8) = *(_DWORD *)(v37 + 8) & 1 | (2 * v27);
        if ( !sub_F34140((__int64)v25, (__int64)v43) )
          --*(_DWORD *)(v37 + 12);
        *v25 = _mm_loadu_si128(&v46);
        v25[1] = _mm_loadu_si128(&v47);
        v25[2].m128i_i64[0] = v48;
        v25[2].m128i_i64[1] = (__int64)&v25[3].m128i_i64[1];
        v25[3].m128i_i64[0] = 0x200000000LL;
        if ( (_DWORD)v50 )
          sub_349D9E0((__int64)&v25[2].m128i_i64[1], &v49, v31, v32, v33, v34);
        goto LABEL_17;
      }
    }
    sub_34A1500(v37, v28, v29, v30, v21, v22);
    sub_34A1150(v37, (__int64)&v46, (__int64 *)&v42);
    v25 = v42;
    v27 = (*(_DWORD *)(v37 + 8) >> 1) + 1;
    goto LABEL_31;
  }
LABEL_17:
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
}
