// Function: sub_265ECC0
// Address: 0x265ecc0
//
unsigned __int64 __fastcall sub_265ECC0(__int64 a1, const char *a2, __int64 *a3, int *a4, _DWORD **a5, __int64 *a6)
{
  size_t v9; // rax
  int *v10; // rcx
  __int64 v11; // rdx
  int v12; // eax
  _DWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  int v16; // edx
  __int64 v17; // rsi
  const __m128i *v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // r9
  __int64 v21; // r8
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r11
  int v26; // esi
  __m128i *v27; // rdx
  __m128i v28; // xmm1
  __int8 v29; // cl
  __int64 v30; // rdi
  char *v32; // rbx
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  _QWORD v39[5]; // [rsp+20h] [rbp-60h] BYREF
  int v40; // [rsp+48h] [rbp-38h]
  char v41; // [rsp+4Ch] [rbp-34h]

  sub_D95050(a1, 0, 0);
  *(_BYTE *)(a1 + 156) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A1FA60;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)a1 = &unk_4A1FAD0;
  *(_QWORD *)(a1 + 160) = &unk_4A1FA80;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1544;
  *(_QWORD *)(a1 + 592) = sub_263DA00;
  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v10 = a4;
  v11 = *a3;
  v36 = a1 + 176;
  *(_QWORD *)(a1 + 48) = a3[1];
  v12 = *v10;
  *(_QWORD *)(a1 + 40) = v11;
  *(_BYTE *)(a1 + 12) = (32 * (v12 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v13 = *a5;
  LODWORD(v11) = **a5;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v11;
  v14 = *((unsigned int *)a6 + 2);
  *(_DWORD *)(a1 + 152) = *v13;
  v15 = *a6;
  v38 = *a6 + 40 * v14;
  if ( *a6 != v38 )
  {
    do
    {
      v16 = *(_DWORD *)(v15 + 16);
      v17 = *(_QWORD *)(v15 + 24);
      v39[4] = &unk_4A1FA60;
      v18 = (const __m128i *)v39;
      v19 = *(_QWORD *)(v15 + 32);
      v20 = *(_QWORD *)v15;
      v41 = 1;
      v21 = *(_QWORD *)(v15 + 8);
      v22 = *(unsigned int *)(a1 + 188);
      v40 = v16;
      v23 = *(unsigned int *)(a1 + 184);
      v39[2] = v17;
      v39[3] = v19;
      v24 = *(_QWORD *)(a1 + 176);
      v25 = v23 + 1;
      v39[0] = v20;
      v26 = v23;
      v39[1] = v21;
      if ( v23 + 1 > v22 )
      {
        if ( v24 > (unsigned __int64)v39 )
        {
          v33 = v21;
          v34 = v20;
LABEL_11:
          sub_265EC00(v36, v25, v23, v24, v21, v20);
          v23 = *(unsigned int *)(a1 + 184);
          v24 = *(_QWORD *)(a1 + 176);
          v20 = v34;
          v21 = v33;
          v26 = *(_DWORD *)(a1 + 184);
          goto LABEL_3;
        }
        v33 = v21;
        v34 = v20;
        v23 = v24 + 48 * v23;
        if ( (unsigned __int64)v39 >= v23 )
          goto LABEL_11;
        v32 = (char *)v39 - v24;
        sub_265EC00(v36, v25, v23, v24, v21, v20);
        v24 = *(_QWORD *)(a1 + 176);
        v23 = *(unsigned int *)(a1 + 184);
        v21 = v33;
        v20 = v34;
        v18 = (const __m128i *)&v32[v24];
        v26 = *(_DWORD *)(a1 + 184);
      }
LABEL_3:
      v27 = (__m128i *)(v24 + 48 * v23);
      if ( v27 )
      {
        v28 = _mm_loadu_si128(v18 + 1);
        *v27 = _mm_loadu_si128(v18);
        v27[1] = v28;
        v27[2].m128i_i32[2] = v18[2].m128i_i32[2];
        v29 = v18[2].m128i_i8[12];
        v27[2].m128i_i64[0] = (__int64)&unk_4A1FA60;
        v27[2].m128i_i8[12] = v29;
        v26 = *(_DWORD *)(a1 + 184);
      }
      v30 = *(_QWORD *)(a1 + 168);
      v15 += 40;
      *(_DWORD *)(a1 + 184) = v26 + 1;
      sub_C52F90(v30, v20, v21);
    }
    while ( v38 != v15 );
  }
  return sub_C53130(a1);
}
