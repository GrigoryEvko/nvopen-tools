// Function: sub_12F11C0
// Address: 0x12f11c0
//
__int64 __fastcall sub_12F11C0(__int64 a1, const char *a2, _QWORD *a3, __int64 **a4)
{
  int v7; // edx
  size_t v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 *v12; // rbx
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r10
  __int64 v19; // r9
  __m128i *v20; // r13
  __int64 v21; // rsi
  __m128i *v22; // rsi
  __int64 v23; // rdi
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  const __m128i *v31; // rdi
  __m128i *v32; // rax
  const __m128i *v33; // r11
  __int8 v34; // dl
  unsigned int v35; // [rsp+Ch] [rbp-74h]
  unsigned int v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  __int64 v40; // [rsp+20h] [rbp-60h]
  __int64 v41; // [rsp+20h] [rbp-60h]
  __int64 v42; // [rsp+28h] [rbp-58h]
  __int64 v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+28h] [rbp-58h]
  int v45; // [rsp+30h] [rbp-50h]
  const __m128i *v46; // [rsp+38h] [rbp-48h]
  __int64 v47; // [rsp+40h] [rbp-40h]
  const __m128i *v48; // [rsp+40h] [rbp-40h]
  __int64 v49; // [rsp+48h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49EED30;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49E7678;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49E76E8;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 184) = &unk_49E7698;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  v46 = (const __m128i *)(a1 + 216);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 192) = a1;
  *(_QWORD *)(a1 + 208) = 0x800000000LL;
  v8 = strlen(a2);
  sub_16B8280(a1, a2, v8);
  v9 = a3[1];
  *(_QWORD *)(a1 + 40) = *a3;
  v10 = *((unsigned int *)a4 + 2);
  *(_QWORD *)(a1 + 48) = v9;
  v11 = (__int64)&(*a4)[5 * v10];
  v12 = *a4;
  v49 = v11;
  while ( (__int64 *)v49 != v12 )
  {
    v13 = *(_DWORD *)(a1 + 208);
    v14 = *(unsigned int *)(a1 + 212);
    v15 = *v12;
    v16 = v12[1];
    v17 = *((unsigned int *)v12 + 4);
    v18 = v12[3];
    v19 = v12[4];
    if ( v13 >= (unsigned int)v14 )
    {
      v36 = *((_DWORD *)v12 + 4);
      v38 = *v12;
      v40 = v12[1];
      v42 = v12[3];
      v47 = v12[4];
      v25 = ((((unsigned __int64)(v14 + 2) >> 1) | (v14 + 2)) >> 2) | ((unsigned __int64)(v14 + 2) >> 1) | (v14 + 2);
      v26 = (((v25 >> 4) | v25) >> 8) | (v25 >> 4) | v25;
      v27 = v26 | (v26 >> 16);
      v28 = (v27 | HIDWORD(v26)) + 1;
      v29 = 0xFFFFFFFFLL;
      if ( v28 <= 0xFFFFFFFF )
        v29 = v28;
      v45 = v29;
      v30 = malloc(48 * v29, v27, v16, v28, v15, v19);
      v19 = v47;
      v18 = v42;
      v16 = v40;
      v15 = v38;
      v20 = (__m128i *)v30;
      v17 = v36;
      if ( !v30 )
      {
        sub_16BD1C0("Allocation failed");
        v13 = *(_DWORD *)(a1 + 208);
        v17 = v36;
        v15 = v38;
        v16 = v40;
        v18 = v42;
        v19 = v47;
      }
      v31 = *(const __m128i **)(a1 + 200);
      v32 = v20;
      v21 = 3LL * v13;
      v48 = v31;
      v33 = &v31[v21];
      if ( v31 != &v31[v21] )
      {
        v43 = v16;
        do
        {
          if ( v32 )
          {
            *v32 = _mm_loadu_si128(v31);
            v32[1] = _mm_loadu_si128(v31 + 1);
            v32[2].m128i_i32[2] = v31[2].m128i_i32[2];
            v34 = v31[2].m128i_i8[12];
            v32[2].m128i_i64[0] = (__int64)&unk_49E7678;
            v32[2].m128i_i8[12] = v34;
          }
          v31 += 3;
          v32 += 3;
        }
        while ( v33 != v31 );
        v16 = v43;
      }
      if ( v46 != v48 )
      {
        v35 = v17;
        v37 = v15;
        v39 = v16;
        v41 = v18;
        v44 = v19;
        _libc_free(v48, v21 * 16);
        v17 = v35;
        v15 = v37;
        v16 = v39;
        v18 = v41;
        v19 = v44;
        v13 = *(_DWORD *)(a1 + 208);
        v21 = 3LL * v13;
      }
      *(_QWORD *)(a1 + 200) = v20;
      *(_DWORD *)(a1 + 212) = v45;
    }
    else
    {
      v20 = *(__m128i **)(a1 + 200);
      v21 = 3LL * v13;
    }
    v22 = &v20[v21];
    if ( v22 )
    {
      v22->m128i_i64[0] = v15;
      v22->m128i_i64[1] = v16;
      v22[1].m128i_i64[0] = v18;
      v22[1].m128i_i64[1] = v19;
      v22[2].m128i_i32[2] = v17;
      v22[2].m128i_i8[12] = 1;
      v22[2].m128i_i64[0] = (__int64)&unk_49E7678;
      v13 = *(_DWORD *)(a1 + 208);
    }
    v23 = *(_QWORD *)(a1 + 192);
    v12 += 5;
    *(_DWORD *)(a1 + 208) = v13 + 1;
    sub_16B7FD0(v23, v15, v16, v17);
  }
  return sub_16B88A0(a1);
}
