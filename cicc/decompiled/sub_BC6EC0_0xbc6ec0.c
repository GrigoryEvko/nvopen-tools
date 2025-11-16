// Function: sub_BC6EC0
// Address: 0xbc6ec0
//
__int64 __fastcall sub_BC6EC0(
        __int64 a1,
        const char *a2,
        __int64 *a3,
        _DWORD *a4,
        _DWORD *a5,
        _DWORD **a6,
        __int64 **a7)
{
  int v9; // edx
  __int64 v10; // rbx
  __int64 v11; // rax
  size_t v12; // rax
  __int64 v13; // rdx
  _DWORD *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r14
  int v17; // edx
  __int64 v18; // rsi
  const __m128i *v19; // r12
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  __int64 v26; // r11
  int v27; // esi
  __m128i *v28; // rdx
  __m128i v29; // xmm1
  __int64 v30; // rdi
  char *v32; // r12
  __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  _QWORD v40[5]; // [rsp+20h] [rbp-60h] BYREF
  int v41; // [rsp+48h] [rbp-38h]
  char v42; // [rsp+4Ch] [rbp-34h]

  *(_QWORD *)a1 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
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
  v10 = sub_C57470();
  v11 = *(unsigned int *)(a1 + 80);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, a1 + 88, v11 + 1, 8);
    v11 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v11) = v10;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_49DB2C8;
  *(_QWORD *)(a1 + 144) = &unk_49DB258;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_49DB278;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_91;
  *(_QWORD *)(a1 + 592) = sub_BC4D50;
  v12 = strlen(a2);
  sub_C53080(a1, a2, v12);
  v13 = *a3;
  *(_QWORD *)(a1 + 48) = a3[1];
  *(_QWORD *)(a1 + 40) = v13;
  LODWORD(v13) = *a5;
  v37 = a1 + 176;
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a1 + 12) & 0x87 | (8 * (v13 & 3)) | (32 * (*a4 & 3));
  v14 = *a6;
  LODWORD(v13) = **a6;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v13;
  v15 = *((unsigned int *)a7 + 2);
  *(_DWORD *)(a1 + 152) = *v14;
  v16 = *a7;
  v39 = (__int64)&(*a7)[5 * v15];
  if ( *a7 != (__int64 *)v39 )
  {
    do
    {
      v17 = *((_DWORD *)v16 + 4);
      v18 = v16[3];
      v40[4] = &unk_49DB258;
      v19 = (const __m128i *)v40;
      v20 = v16[4];
      v21 = *v16;
      v42 = 1;
      v22 = v16[1];
      v23 = *(unsigned int *)(a1 + 188);
      v41 = v17;
      v24 = *(unsigned int *)(a1 + 184);
      v40[2] = v18;
      v40[3] = v20;
      v25 = *(_QWORD *)(a1 + 176);
      v26 = v24 + 1;
      v40[0] = v21;
      v27 = v24;
      v40[1] = v22;
      if ( v24 + 1 > v23 )
      {
        if ( v25 > (unsigned __int64)v40 )
        {
          v33 = v22;
          v35 = v21;
LABEL_13:
          sub_BC6E00(v37, v26);
          v24 = *(unsigned int *)(a1 + 184);
          v25 = *(_QWORD *)(a1 + 176);
          v21 = v35;
          v22 = v33;
          v27 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v33 = v22;
        v35 = v21;
        if ( (unsigned __int64)v40 >= v25 + 48 * v24 )
          goto LABEL_13;
        v32 = (char *)v40 - v25;
        sub_BC6E00(v37, v26);
        v25 = *(_QWORD *)(a1 + 176);
        v24 = *(unsigned int *)(a1 + 184);
        v22 = v33;
        v21 = v35;
        v19 = (const __m128i *)&v32[v25];
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
        v25 = v19[2].m128i_u8[12];
        v28[2].m128i_i64[0] = (__int64)&unk_49DB258;
        v28[2].m128i_i8[12] = v25;
        v27 = *(_DWORD *)(a1 + 184);
      }
      v30 = *(_QWORD *)(a1 + 168);
      v16 += 5;
      *(_DWORD *)(a1 + 184) = v27 + 1;
      sub_C52F90(v30, v21, v22, v25);
    }
    while ( (__int64 *)v39 != v16 );
  }
  return sub_C53130(a1);
}
