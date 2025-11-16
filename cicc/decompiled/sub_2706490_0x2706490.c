// Function: sub_2706490
// Address: 0x2706490
//
unsigned __int64 __fastcall sub_2706490(__int64 a1, const char *a2, _QWORD *a3, __int64 *a4, _BYTE *a5)
{
  __int64 v7; // r12
  int v8; // edx
  __int64 *v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  size_t v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  int v19; // edx
  __int64 v20; // r10
  const __m128i *v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // r9
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
  __int64 v36; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  __int64 v39; // [rsp+20h] [rbp-70h]
  __int64 v40; // [rsp+28h] [rbp-68h]
  _QWORD v41[5]; // [rsp+30h] [rbp-60h] BYREF
  int v42; // [rsp+58h] [rbp-38h]
  char v43; // [rsp+5Ch] [rbp-34h]

  v7 = a1;
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
  v9 = sub_C57470();
  v12 = *(unsigned int *)(a1 + 80);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v10, v11);
    v12 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v12) = v9;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A1F880;
  *(_QWORD *)a1 = &unk_4A1F8F0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A1F8A0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1542;
  *(_QWORD *)(a1 + 592) = sub_2618AB0;
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  v15 = a3[1];
  v39 = a1 + 176;
  *(_QWORD *)(a1 + 40) = *a3;
  v16 = *((unsigned int *)a4 + 2);
  *(_QWORD *)(a1 + 48) = v15;
  if ( *a4 != *a4 + 40 * v16 )
  {
    v40 = *a4 + 40 * v16;
    v17 = *a4;
    while ( 1 )
    {
      v19 = *(_DWORD *)(v17 + 16);
      v20 = *(_QWORD *)v17;
      v41[4] = &unk_4A1F880;
      v21 = (const __m128i *)v41;
      v22 = *(_QWORD *)(v17 + 24);
      v23 = *(_QWORD *)(v17 + 32);
      v43 = 1;
      v24 = *(_QWORD *)(v17 + 8);
      v25 = *(unsigned int *)(a1 + 188);
      v42 = v19;
      v26 = *(unsigned int *)(a1 + 184);
      v41[2] = v22;
      v41[3] = v23;
      v27 = *(_QWORD *)(a1 + 176);
      v28 = v26 + 1;
      v41[0] = v20;
      v29 = v26;
      v41[1] = v24;
      if ( v26 + 1 > v25 )
      {
        if ( v27 > (unsigned __int64)v41 )
        {
          v36 = v24;
          v37 = v20;
LABEL_15:
          sub_263C100(v39, v28, v26, v27, v14, v24);
          v26 = *(unsigned int *)(a1 + 184);
          v20 = v37;
          v27 = *(_QWORD *)(a1 + 176);
          v24 = v36;
          v29 = *(_DWORD *)(a1 + 184);
          goto LABEL_6;
        }
        v36 = v24;
        v37 = v20;
        v26 = v27 + 48 * v26;
        if ( (unsigned __int64)v41 >= v26 )
          goto LABEL_15;
        v35 = (char *)v41 - v27;
        sub_263C100(v39, v28, v26, v27, v14, v24);
        v27 = *(_QWORD *)(a1 + 176);
        v26 = *(unsigned int *)(a1 + 184);
        v24 = v36;
        v20 = v37;
        v21 = (const __m128i *)&v35[v27];
        v29 = *(_DWORD *)(a1 + 184);
      }
LABEL_6:
      v30 = (__m128i *)(v27 + 48 * v26);
      if ( v30 )
      {
        v31 = _mm_loadu_si128(v21 + 1);
        *v30 = _mm_loadu_si128(v21);
        v30[1] = v31;
        v30[2].m128i_i32[2] = v21[2].m128i_i32[2];
        v32 = v21[2].m128i_i8[12];
        v30[2].m128i_i64[0] = (__int64)&unk_4A1F880;
        v30[2].m128i_i8[12] = v32;
        v29 = *(_DWORD *)(a1 + 184);
      }
      v33 = *(_QWORD *)(a1 + 168);
      v17 += 40;
      *(_DWORD *)(a1 + 184) = v29 + 1;
      sub_C52F90(v33, v20, v24);
      if ( v40 == v17 )
      {
        v7 = a1;
        break;
      }
    }
  }
  *(_BYTE *)(v7 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(v7 + 12) & 0x9F;
  return sub_C53130(v7);
}
