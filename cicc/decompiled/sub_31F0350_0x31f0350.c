// Function: sub_31F0350
// Address: 0x31f0350
//
unsigned __int64 __fastcall sub_31F0350(__int64 a1, const char *a2, _DWORD *a3, _BYTE *a4, __int64 *a5, __int64 *a6)
{
  __int64 v8; // r12
  int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // rdx
  size_t v14; // rax
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // r12
  int v20; // edx
  __int64 v21; // r10
  const __m128i *v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r9
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // r11
  int v30; // esi
  __m128i *v31; // rdx
  __m128i v32; // xmm1
  __int8 v33; // cl
  __int64 v34; // rdi
  __int64 v35; // rdx
  char *v37; // rbx
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v41; // [rsp+20h] [rbp-70h]
  __int64 *v42; // [rsp+28h] [rbp-68h]
  _QWORD v43[5]; // [rsp+30h] [rbp-60h] BYREF
  int v44; // [rsp+58h] [rbp-38h]
  char v45; // [rsp+5Ch] [rbp-34h]

  v8 = a1;
  *(_QWORD *)a1 = &unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a1 + 12) & 0x8000 | 1;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v11 = sub_C57470();
  v13 = *(unsigned int *)(a1 + 80);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v42 = v11;
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v13 + 1, 8u, v13 + 1, v12);
    v13 = *(unsigned int *)(a1 + 80);
    v11 = v42;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v13) = v11;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)a1 = &unk_4A34FB8;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 176) = a1;
  *(_QWORD *)(a1 + 168) = &unk_4A34F68;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  *(_QWORD *)(a1 + 192) = 0x800000000LL;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 608) = nullsub_1847;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 600) = sub_31D49F0;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v16 = *((unsigned int *)a5 + 2);
  *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xC19F
                      | (32 * (*a3 & 3))
                      | (((*a4 | (*(_BYTE *)(a1 + 13) >> 1)) & 0x1F) << 9);
  v17 = *a5 + 40 * v16;
  if ( *a5 != v17 )
  {
    v18 = *a5;
    v41 = a1 + 184;
    while ( 1 )
    {
      v20 = *(_DWORD *)(v18 + 16);
      v21 = *(_QWORD *)v18;
      v43[4] = &unk_4A34F48;
      v22 = (const __m128i *)v43;
      v23 = *(_QWORD *)(v18 + 24);
      v24 = *(_QWORD *)(v18 + 32);
      v45 = 1;
      v25 = *(_QWORD *)(v18 + 8);
      v26 = *(unsigned int *)(a1 + 196);
      v44 = v20;
      v27 = *(unsigned int *)(a1 + 192);
      v43[2] = v23;
      v43[3] = v24;
      v28 = *(_QWORD *)(a1 + 184);
      v29 = v27 + 1;
      v43[0] = v21;
      v30 = v27;
      v43[1] = v25;
      if ( v27 + 1 > v26 )
      {
        if ( v28 > (unsigned __int64)v43 )
        {
          v38 = v25;
          v39 = v21;
LABEL_15:
          sub_31F0290(v41, v29, v27, v28, v15, v25);
          v27 = *(unsigned int *)(a1 + 192);
          v21 = v39;
          v28 = *(_QWORD *)(a1 + 184);
          v25 = v38;
          v30 = *(_DWORD *)(a1 + 192);
          goto LABEL_6;
        }
        v38 = v25;
        v39 = v21;
        v27 = v28 + 48 * v27;
        if ( (unsigned __int64)v43 >= v27 )
          goto LABEL_15;
        v37 = (char *)v43 - v28;
        sub_31F0290(v41, v29, v27, v28, v15, v25);
        v28 = *(_QWORD *)(a1 + 184);
        v27 = *(unsigned int *)(a1 + 192);
        v25 = v38;
        v21 = v39;
        v22 = (const __m128i *)&v37[v28];
        v30 = *(_DWORD *)(a1 + 192);
      }
LABEL_6:
      v31 = (__m128i *)(v28 + 48 * v27);
      if ( v31 )
      {
        v32 = _mm_loadu_si128(v22 + 1);
        *v31 = _mm_loadu_si128(v22);
        v31[1] = v32;
        v31[2].m128i_i32[2] = v22[2].m128i_i32[2];
        v33 = v22[2].m128i_i8[12];
        v31[2].m128i_i64[0] = (__int64)&unk_4A34F48;
        v31[2].m128i_i8[12] = v33;
        v30 = *(_DWORD *)(a1 + 192);
      }
      v34 = *(_QWORD *)(a1 + 176);
      v18 += 40;
      *(_DWORD *)(a1 + 192) = v30 + 1;
      sub_C52F90(v34, v21, v25);
      if ( v17 == v18 )
      {
        v8 = a1;
        break;
      }
    }
  }
  v35 = *a6;
  *(_QWORD *)(v8 + 48) = a6[1];
  *(_QWORD *)(v8 + 40) = v35;
  return sub_C53130(v8);
}
