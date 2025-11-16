// Function: sub_2EA3C20
// Address: 0x2ea3c20
//
unsigned __int64 __fastcall sub_2EA3C20(__int64 a1, const char *a2, __int64 *a3, _DWORD **a4, _DWORD *a5, __int64 *a6)
{
  int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // r12
  __int64 v13; // rax
  size_t v14; // rax
  __int64 v15; // rdx
  _DWORD *v16; // rax
  _DWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  int v20; // edx
  __int64 v21; // rsi
  const __m128i *v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 v25; // r8
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // r11
  int v30; // esi
  __m128i *v31; // rdx
  __m128i v32; // xmm1
  __int8 v33; // cl
  __int64 v34; // rdi
  char *v36; // rbx
  __int64 v37; // [rsp+0h] [rbp-80h]
  __int64 v38; // [rsp+8h] [rbp-78h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  _QWORD v43[5]; // [rsp+20h] [rbp-60h] BYREF
  int v44; // [rsp+48h] [rbp-38h]
  char v45; // [rsp+4Ch] [rbp-34h]

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
  *(_QWORD *)a1 = &unk_4A29348;
  *(_QWORD *)(a1 + 144) = &unk_4A292D8;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A292F8;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1604;
  *(_QWORD *)(a1 + 592) = sub_2E97430;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v15 = *a3;
  *(_QWORD *)(a1 + 48) = a3[1];
  *(_QWORD *)(a1 + 40) = v15;
  v16 = *a4;
  LODWORD(v15) = **a4;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v15;
  *(_DWORD *)(a1 + 152) = *v16;
  v17 = a5;
  v40 = a1 + 176;
  v18 = *((unsigned int *)a6 + 2);
  *(_BYTE *)(a1 + 12) = (32 * (*v17 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v19 = *a6;
  v42 = *a6 + 40 * v18;
  if ( *a6 != v42 )
  {
    do
    {
      v20 = *(_DWORD *)(v19 + 16);
      v21 = *(_QWORD *)(v19 + 24);
      v43[4] = &unk_4A292D8;
      v22 = (const __m128i *)v43;
      v23 = *(_QWORD *)(v19 + 32);
      v24 = *(_QWORD *)v19;
      v45 = 1;
      v25 = *(_QWORD *)(v19 + 8);
      v26 = *(unsigned int *)(a1 + 188);
      v44 = v20;
      v27 = *(unsigned int *)(a1 + 184);
      v43[2] = v21;
      v43[3] = v23;
      v28 = *(_QWORD *)(a1 + 176);
      v29 = v27 + 1;
      v43[0] = v24;
      v30 = v27;
      v43[1] = v25;
      if ( v27 + 1 > v26 )
      {
        if ( v28 > (unsigned __int64)v43 )
        {
          v37 = v25;
          v38 = v24;
LABEL_13:
          sub_2EA3B60(v40, v29, v27, v28, v25, v24);
          v27 = *(unsigned int *)(a1 + 184);
          v28 = *(_QWORD *)(a1 + 176);
          v24 = v38;
          v25 = v37;
          v30 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v37 = v25;
        v38 = v24;
        v27 = v28 + 48 * v27;
        if ( (unsigned __int64)v43 >= v27 )
          goto LABEL_13;
        v36 = (char *)v43 - v28;
        sub_2EA3B60(v40, v29, v27, v28, v25, v24);
        v28 = *(_QWORD *)(a1 + 176);
        v27 = *(unsigned int *)(a1 + 184);
        v25 = v37;
        v24 = v38;
        v22 = (const __m128i *)&v36[v28];
        v30 = *(_DWORD *)(a1 + 184);
      }
LABEL_5:
      v31 = (__m128i *)(v28 + 48 * v27);
      if ( v31 )
      {
        v32 = _mm_loadu_si128(v22 + 1);
        *v31 = _mm_loadu_si128(v22);
        v31[1] = v32;
        v31[2].m128i_i32[2] = v22[2].m128i_i32[2];
        v33 = v22[2].m128i_i8[12];
        v31[2].m128i_i64[0] = (__int64)&unk_4A292D8;
        v31[2].m128i_i8[12] = v33;
        v30 = *(_DWORD *)(a1 + 184);
      }
      v34 = *(_QWORD *)(a1 + 168);
      v19 += 40;
      *(_DWORD *)(a1 + 184) = v30 + 1;
      sub_C52F90(v34, v24, v25);
    }
    while ( v42 != v19 );
  }
  return sub_C53130(a1);
}
