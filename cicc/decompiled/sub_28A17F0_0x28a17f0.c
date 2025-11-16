// Function: sub_28A17F0
// Address: 0x28a17f0
//
unsigned __int64 __fastcall sub_28A17F0(__int64 a1, const char *a2, int **a3, __int64 *a4, __int64 *a5)
{
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r12
  __int64 v12; // rax
  size_t v13; // rax
  __int64 v14; // r9
  int *v15; // rax
  int v16; // edx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  int v21; // edx
  __int64 v22; // rsi
  const __m128i *v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // r10
  __int64 v26; // r8
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // r11
  int v31; // esi
  __m128i *v32; // rdx
  __m128i v33; // xmm1
  __int8 v34; // cl
  __int64 v35; // rdi
  char *v37; // rbx
  __int64 v38; // [rsp+0h] [rbp-80h]
  __int64 v39; // [rsp+8h] [rbp-78h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  _QWORD v43[5]; // [rsp+20h] [rbp-60h] BYREF
  int v44; // [rsp+48h] [rbp-38h]
  char v45; // [rsp+4Ch] [rbp-34h]

  *(_QWORD *)a1 = &unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v8;
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
  v12 = *(unsigned int *)(a1 + 80);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v9, v10);
    v12 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v12) = v11;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A21718;
  *(_QWORD *)(a1 + 144) = &unk_4A216A8;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A216C8;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1554;
  *(_QWORD *)(a1 + 592) = sub_28940E0;
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  v40 = a1 + 176;
  v15 = *a3;
  v16 = **a3;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v16;
  v17 = *a4;
  *(_DWORD *)(a1 + 152) = *v15;
  v18 = a4[1];
  *(_QWORD *)(a1 + 40) = v17;
  v19 = *((unsigned int *)a5 + 2);
  *(_QWORD *)(a1 + 48) = v18;
  v20 = *a5;
  v42 = *a5 + 40 * v19;
  if ( *a5 != v42 )
  {
    do
    {
      v21 = *(_DWORD *)(v20 + 16);
      v22 = *(_QWORD *)(v20 + 24);
      v43[4] = &unk_4A216A8;
      v23 = (const __m128i *)v43;
      v24 = *(_QWORD *)(v20 + 32);
      v25 = *(_QWORD *)v20;
      v45 = 1;
      v26 = *(_QWORD *)(v20 + 8);
      v27 = *(unsigned int *)(a1 + 188);
      v44 = v21;
      v28 = *(unsigned int *)(a1 + 184);
      v43[2] = v22;
      v43[3] = v24;
      v29 = *(_QWORD *)(a1 + 176);
      v30 = v28 + 1;
      v43[0] = v25;
      v31 = v28;
      v43[1] = v26;
      if ( v28 + 1 > v27 )
      {
        if ( v29 > (unsigned __int64)v43 )
        {
          v38 = v26;
          v39 = v25;
LABEL_13:
          sub_28A1730(v40, v30, v28, v29, v26, v14);
          v28 = *(unsigned int *)(a1 + 184);
          v29 = *(_QWORD *)(a1 + 176);
          v25 = v39;
          v26 = v38;
          v31 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v38 = v26;
        v39 = v25;
        v28 = v29 + 48 * v28;
        if ( (unsigned __int64)v43 >= v28 )
          goto LABEL_13;
        v37 = (char *)v43 - v29;
        sub_28A1730(v40, v30, v28, v29, v26, v14);
        v29 = *(_QWORD *)(a1 + 176);
        v28 = *(unsigned int *)(a1 + 184);
        v26 = v38;
        v25 = v39;
        v23 = (const __m128i *)&v37[v29];
        v31 = *(_DWORD *)(a1 + 184);
      }
LABEL_5:
      v32 = (__m128i *)(v29 + 48 * v28);
      if ( v32 )
      {
        v33 = _mm_loadu_si128(v23 + 1);
        *v32 = _mm_loadu_si128(v23);
        v32[1] = v33;
        v32[2].m128i_i32[2] = v23[2].m128i_i32[2];
        v34 = v23[2].m128i_i8[12];
        v32[2].m128i_i64[0] = (__int64)&unk_4A216A8;
        v32[2].m128i_i8[12] = v34;
        v31 = *(_DWORD *)(a1 + 184);
      }
      v35 = *(_QWORD *)(a1 + 168);
      v20 += 40;
      *(_DWORD *)(a1 + 184) = v31 + 1;
      sub_C52F90(v35, v25, v26);
    }
    while ( v42 != v20 );
  }
  return sub_C53130(a1);
}
