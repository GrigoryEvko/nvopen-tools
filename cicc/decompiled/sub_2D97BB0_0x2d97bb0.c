// Function: sub_2D97BB0
// Address: 0x2d97bb0
//
unsigned __int64 __fastcall sub_2D97BB0(__int64 a1, const char *a2, __int64 *a3, _DWORD **a4, __int64 *a5)
{
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r12
  __int64 v12; // rax
  size_t v13; // rax
  __int64 v14; // r9
  __int64 v15; // rdx
  _DWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  int v19; // edx
  __int64 v20; // rsi
  const __m128i *v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // r10
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // r11
  int v29; // esi
  __m128i *v30; // rdx
  __m128i v31; // xmm1
  __int8 v32; // cl
  __int64 v33; // rdi
  char *v35; // rbx
  __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  _QWORD v41[5]; // [rsp+20h] [rbp-60h] BYREF
  int v42; // [rsp+48h] [rbp-38h]
  char v43; // [rsp+4Ch] [rbp-34h]

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
  *(_QWORD *)a1 = &unk_4A27120;
  *(_QWORD *)(a1 + 144) = &unk_4A270B0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A270D0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1595;
  *(_QWORD *)(a1 + 592) = sub_2D8CCF0;
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  v15 = *a3;
  v38 = a1 + 176;
  *(_QWORD *)(a1 + 48) = a3[1];
  *(_QWORD *)(a1 + 40) = v15;
  v16 = *a4;
  LODWORD(v15) = **a4;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v15;
  v17 = *((unsigned int *)a5 + 2);
  *(_DWORD *)(a1 + 152) = *v16;
  v18 = *a5;
  v40 = *a5 + 40 * v17;
  if ( *a5 != v40 )
  {
    do
    {
      v19 = *(_DWORD *)(v18 + 16);
      v20 = *(_QWORD *)(v18 + 24);
      v41[4] = &unk_4A270B0;
      v21 = (const __m128i *)v41;
      v22 = *(_QWORD *)(v18 + 32);
      v23 = *(_QWORD *)v18;
      v43 = 1;
      v24 = *(_QWORD *)(v18 + 8);
      v25 = *(unsigned int *)(a1 + 188);
      v42 = v19;
      v26 = *(unsigned int *)(a1 + 184);
      v41[2] = v20;
      v41[3] = v22;
      v27 = *(_QWORD *)(a1 + 176);
      v28 = v26 + 1;
      v41[0] = v23;
      v29 = v26;
      v41[1] = v24;
      if ( v26 + 1 > v25 )
      {
        if ( v27 > (unsigned __int64)v41 )
        {
          v36 = v24;
          v37 = v23;
LABEL_13:
          sub_2D97AF0(v38, v28, v26, v27, v24, v14);
          v26 = *(unsigned int *)(a1 + 184);
          v27 = *(_QWORD *)(a1 + 176);
          v23 = v37;
          v24 = v36;
          v29 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v36 = v24;
        v37 = v23;
        v26 = v27 + 48 * v26;
        if ( (unsigned __int64)v41 >= v26 )
          goto LABEL_13;
        v35 = (char *)v41 - v27;
        sub_2D97AF0(v38, v28, v26, v27, v24, v14);
        v27 = *(_QWORD *)(a1 + 176);
        v26 = *(unsigned int *)(a1 + 184);
        v24 = v36;
        v23 = v37;
        v21 = (const __m128i *)&v35[v27];
        v29 = *(_DWORD *)(a1 + 184);
      }
LABEL_5:
      v30 = (__m128i *)(v27 + 48 * v26);
      if ( v30 )
      {
        v31 = _mm_loadu_si128(v21 + 1);
        *v30 = _mm_loadu_si128(v21);
        v30[1] = v31;
        v30[2].m128i_i32[2] = v21[2].m128i_i32[2];
        v32 = v21[2].m128i_i8[12];
        v30[2].m128i_i64[0] = (__int64)&unk_4A270B0;
        v30[2].m128i_i8[12] = v32;
        v29 = *(_DWORD *)(a1 + 184);
      }
      v33 = *(_QWORD *)(a1 + 168);
      v18 += 40;
      *(_DWORD *)(a1 + 184) = v29 + 1;
      sub_C52F90(v33, v23, v24);
    }
    while ( v40 != v18 );
  }
  return sub_C53130(a1);
}
