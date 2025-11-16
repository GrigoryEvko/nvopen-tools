// Function: sub_2ED0E10
// Address: 0x2ed0e10
//
unsigned __int64 __fastcall sub_2ED0E10(__int64 a1, const char *a2, _DWORD *a3, __int64 *a4, _DWORD **a5, __int64 *a6)
{
  int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // r12
  __int64 v13; // rax
  size_t v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  _DWORD **v17; // rax
  _DWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  int v21; // edx
  __int64 v22; // rsi
  const __m128i *v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // r9
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
  __int64 v41; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  _QWORD v44[5]; // [rsp+20h] [rbp-60h] BYREF
  int v45; // [rsp+48h] [rbp-38h]
  char v46; // [rsp+4Ch] [rbp-34h]

  *(_QWORD *)a1 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v9;
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
  v12 = sub_C57470();
  v13 = *(unsigned int *)(a1 + 80);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v13 + 1, 8u, v10, v11);
    v13 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v13) = v12;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A299F0;
  *(_QWORD *)(a1 + 144) = &unk_4A29980;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A299A0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1626;
  *(_QWORD *)(a1 + 592) = sub_2EC0C60;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v15 = *a4;
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v16 = a4[1];
  *(_QWORD *)(a1 + 40) = v15;
  *(_QWORD *)(a1 + 48) = v16;
  v17 = a5;
  v41 = a1 + 176;
  v18 = *v17;
  LODWORD(v15) = *v18;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v15;
  v19 = *((unsigned int *)a6 + 2);
  *(_DWORD *)(a1 + 152) = *v18;
  v20 = *a6;
  v43 = *a6 + 40 * v19;
  if ( *a6 != v43 )
  {
    do
    {
      v21 = *(_DWORD *)(v20 + 16);
      v22 = *(_QWORD *)(v20 + 24);
      v44[4] = &unk_4A29980;
      v23 = (const __m128i *)v44;
      v24 = *(_QWORD *)(v20 + 32);
      v25 = *(_QWORD *)v20;
      v46 = 1;
      v26 = *(_QWORD *)(v20 + 8);
      v27 = *(unsigned int *)(a1 + 188);
      v45 = v21;
      v28 = *(unsigned int *)(a1 + 184);
      v44[2] = v22;
      v44[3] = v24;
      v29 = *(_QWORD *)(a1 + 176);
      v30 = v28 + 1;
      v44[0] = v25;
      v31 = v28;
      v44[1] = v26;
      if ( v28 + 1 > v27 )
      {
        if ( v29 > (unsigned __int64)v44 )
        {
          v38 = v26;
          v39 = v25;
LABEL_13:
          sub_2ED09C0(v41, v30, v28, v29, v26, v25);
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
        if ( (unsigned __int64)v44 >= v28 )
          goto LABEL_13;
        v37 = (char *)v44 - v29;
        sub_2ED09C0(v41, v30, v28, v29, v26, v25);
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
        v32[2].m128i_i64[0] = (__int64)&unk_4A29980;
        v32[2].m128i_i8[12] = v34;
        v31 = *(_DWORD *)(a1 + 184);
      }
      v35 = *(_QWORD *)(a1 + 168);
      v20 += 40;
      *(_DWORD *)(a1 + 184) = v31 + 1;
      sub_C52F90(v35, v25, v26);
    }
    while ( v43 != v20 );
  }
  return sub_C53130(a1);
}
