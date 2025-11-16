// Function: sub_2D94AB0
// Address: 0x2d94ab0
//
unsigned __int64 __fastcall sub_2D94AB0(__int64 a1, const char *a2, _QWORD *a3, __int64 *a4)
{
  int v7; // edx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // r12
  __int64 v11; // rax
  size_t v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  int v17; // edx
  __int64 v18; // rsi
  const __m128i *v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // r9
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // r11
  int v27; // esi
  __m128i *v28; // rdx
  __m128i v29; // xmm1
  __int8 v30; // cl
  __int64 v31; // rdi
  char *v33; // rbx
  __int64 v34; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  _QWORD v38[5]; // [rsp+20h] [rbp-60h] BYREF
  int v39; // [rsp+48h] [rbp-38h]
  char v40; // [rsp+4Ch] [rbp-34h]

  *(_QWORD *)a1 = &unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v7;
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
  v10 = sub_C57470();
  v11 = *(unsigned int *)(a1 + 80);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v11 + 1, 8u, v8, v9);
    v11 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v11) = v10;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A266D0;
  *(_QWORD *)(a1 + 144) = &unk_4A26660;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A26680;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1584;
  *(_QWORD *)(a1 + 592) = sub_2D8CB90;
  v12 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v12);
  v14 = a3[1];
  v36 = a1 + 176;
  *(_QWORD *)(a1 + 40) = *a3;
  v15 = *((unsigned int *)a4 + 2);
  *(_QWORD *)(a1 + 48) = v14;
  v16 = *a4;
  v37 = *a4 + 40 * v15;
  if ( *a4 != v37 )
  {
    do
    {
      v17 = *(_DWORD *)(v16 + 16);
      v18 = *(_QWORD *)(v16 + 24);
      v38[4] = &unk_4A26660;
      v19 = (const __m128i *)v38;
      v20 = *(_QWORD *)(v16 + 32);
      v21 = *(_QWORD *)v16;
      v40 = 1;
      v22 = *(_QWORD *)(v16 + 8);
      v23 = *(unsigned int *)(a1 + 188);
      v39 = v17;
      v24 = *(unsigned int *)(a1 + 184);
      v38[2] = v18;
      v38[3] = v20;
      v25 = *(_QWORD *)(a1 + 176);
      v26 = v24 + 1;
      v38[0] = v21;
      v27 = v24;
      v38[1] = v22;
      if ( v24 + 1 > v23 )
      {
        if ( v25 > (unsigned __int64)v38 )
        {
          v34 = v22;
          v35 = v21;
LABEL_13:
          sub_2D949F0(v36, v26, v24, v25, v13, v22);
          v24 = *(unsigned int *)(a1 + 184);
          v25 = *(_QWORD *)(a1 + 176);
          v21 = v35;
          v22 = v34;
          v27 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v34 = v22;
        v35 = v21;
        v24 = v25 + 48 * v24;
        if ( (unsigned __int64)v38 >= v24 )
          goto LABEL_13;
        v33 = (char *)v38 - v25;
        sub_2D949F0(v36, v26, v24, v25, v13, v22);
        v25 = *(_QWORD *)(a1 + 176);
        v24 = *(unsigned int *)(a1 + 184);
        v22 = v34;
        v21 = v35;
        v19 = (const __m128i *)&v33[v25];
        v27 = *(_DWORD *)(a1 + 184);
      }
LABEL_5:
      v28 = (__m128i *)(v25 + 48 * v24);
      if ( v28 )
      {
        v29 = _mm_loadu_si128(v19 + 1);
        *v28 = _mm_loadu_si128(v19);
        v28[1] = v29;
        v28[2].m128i_i32[2] = v19[2].m128i_i32[2];
        v30 = v19[2].m128i_i8[12];
        v28[2].m128i_i64[0] = (__int64)&unk_4A26660;
        v28[2].m128i_i8[12] = v30;
        v27 = *(_DWORD *)(a1 + 184);
      }
      v31 = *(_QWORD *)(a1 + 168);
      v16 += 40;
      *(_DWORD *)(a1 + 184) = v27 + 1;
      sub_C52F90(v31, v21, v22);
    }
    while ( v37 != v16 );
  }
  return sub_C53130(a1);
}
