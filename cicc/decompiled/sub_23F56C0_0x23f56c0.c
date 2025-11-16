// Function: sub_23F56C0
// Address: 0x23f56c0
//
unsigned __int64 __fastcall sub_23F56C0(__int64 a1, const char *a2, _QWORD *a3, __int64 *a4, _BYTE *a5, int **a6)
{
  __int64 v6; // r15
  int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // r12
  __int64 v13; // rax
  size_t v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r15
  int v20; // edx
  __int64 v21; // rsi
  const __m128i *v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // r10
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
  int *v35; // rax
  int v36; // edx
  char *v38; // rbx
  __int64 v39; // [rsp+0h] [rbp-90h]
  __int64 v40; // [rsp+8h] [rbp-88h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  _QWORD v45[5]; // [rsp+30h] [rbp-60h] BYREF
  int v46; // [rsp+58h] [rbp-38h]
  char v47; // [rsp+5Ch] [rbp-34h]

  v6 = a1;
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
  *(_QWORD *)a1 = &unk_4A16488;
  *(_QWORD *)(a1 + 144) = &unk_4A16418;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A16438;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1498;
  *(_QWORD *)(a1 + 592) = sub_23DBC70;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v16 = a3[1];
  v43 = a1 + 176;
  *(_QWORD *)(a1 + 40) = *a3;
  v17 = *((unsigned int *)a4 + 2);
  *(_QWORD *)(a1 + 48) = v16;
  v44 = *a4 + 40 * v17;
  if ( *a4 != v44 )
  {
    v18 = *a4;
    while ( 1 )
    {
      v20 = *(_DWORD *)(v18 + 16);
      v21 = *(_QWORD *)(v18 + 24);
      v45[4] = &unk_4A16418;
      v22 = (const __m128i *)v45;
      v23 = *(_QWORD *)(v18 + 32);
      v24 = *(_QWORD *)v18;
      v47 = 1;
      v25 = *(_QWORD *)(v18 + 8);
      v26 = *(unsigned int *)(a1 + 188);
      v46 = v20;
      v27 = *(unsigned int *)(a1 + 184);
      v45[2] = v21;
      v45[3] = v23;
      v28 = *(_QWORD *)(a1 + 176);
      v29 = v27 + 1;
      v45[0] = v24;
      v30 = v27;
      v45[1] = v25;
      if ( v27 + 1 > v26 )
      {
        if ( v28 > (unsigned __int64)v45 )
        {
          v39 = v25;
          v40 = v24;
LABEL_15:
          sub_23F5600(v43, v29, v27, v28, v15, v25);
          v27 = *(unsigned int *)(a1 + 184);
          v28 = *(_QWORD *)(a1 + 176);
          v24 = v40;
          v25 = v39;
          v30 = *(_DWORD *)(a1 + 184);
          goto LABEL_6;
        }
        v39 = v25;
        v40 = v24;
        v27 = v28 + 48 * v27;
        if ( (unsigned __int64)v45 >= v27 )
          goto LABEL_15;
        v38 = (char *)v45 - v28;
        sub_23F5600(v43, v29, v27, v28, v15, v25);
        v28 = *(_QWORD *)(a1 + 176);
        v27 = *(unsigned int *)(a1 + 184);
        v25 = v39;
        v24 = v40;
        v22 = (const __m128i *)&v38[v28];
        v30 = *(_DWORD *)(a1 + 184);
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
        v31[2].m128i_i64[0] = (__int64)&unk_4A16418;
        v31[2].m128i_i8[12] = v33;
        v30 = *(_DWORD *)(a1 + 184);
      }
      v34 = *(_QWORD *)(a1 + 168);
      v18 += 40;
      *(_DWORD *)(a1 + 184) = v30 + 1;
      sub_C52F90(v34, v24, v25);
      if ( v44 == v18 )
      {
        v6 = a1;
        break;
      }
    }
  }
  *(_BYTE *)(v6 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(v6 + 12) & 0x9F;
  v35 = *a6;
  v36 = **a6;
  *(_BYTE *)(v6 + 156) = 1;
  *(_DWORD *)(v6 + 136) = v36;
  *(_DWORD *)(v6 + 152) = *v35;
  return sub_C53130(v6);
}
