// Function: sub_2AF6330
// Address: 0x2af6330
//
unsigned __int64 __fastcall sub_2AF6330(__int64 a1, const char *a2, __int64 *a3, _DWORD **a4, __int64 *a5)
{
  size_t v8; // rax
  __int64 v9; // r9
  __int64 v10; // rdx
  _DWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  int v14; // edx
  __int64 v15; // rsi
  const __m128i *v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // r10
  __int64 v19; // r8
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r11
  int v24; // esi
  __m128i *v25; // rdx
  __m128i v26; // xmm1
  __int8 v27; // cl
  __int64 v28; // rdi
  char *v30; // rbx
  __int64 v31; // [rsp+0h] [rbp-80h]
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  _QWORD v36[5]; // [rsp+20h] [rbp-60h] BYREF
  int v37; // [rsp+48h] [rbp-38h]
  char v38; // [rsp+4Ch] [rbp-34h]

  sub_D95050(a1, 0, 0);
  *(_BYTE *)(a1 + 156) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A23530;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)a1 = &unk_4A235A0;
  *(_QWORD *)(a1 + 160) = &unk_4A23550;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1570;
  *(_QWORD *)(a1 + 592) = sub_2AA7730;
  v8 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v8);
  v10 = *a3;
  v33 = a1 + 176;
  *(_QWORD *)(a1 + 48) = a3[1];
  v11 = *a4;
  *(_QWORD *)(a1 + 40) = v10;
  LODWORD(v10) = *v11;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v10;
  v12 = *((unsigned int *)a5 + 2);
  *(_DWORD *)(a1 + 152) = *v11;
  v13 = *a5;
  v35 = *a5 + 40 * v12;
  if ( *a5 != v35 )
  {
    do
    {
      v14 = *(_DWORD *)(v13 + 16);
      v15 = *(_QWORD *)(v13 + 24);
      v36[4] = &unk_4A23530;
      v16 = (const __m128i *)v36;
      v17 = *(_QWORD *)(v13 + 32);
      v18 = *(_QWORD *)v13;
      v38 = 1;
      v19 = *(_QWORD *)(v13 + 8);
      v20 = *(unsigned int *)(a1 + 188);
      v37 = v14;
      v21 = *(unsigned int *)(a1 + 184);
      v36[2] = v15;
      v36[3] = v17;
      v22 = *(_QWORD *)(a1 + 176);
      v23 = v21 + 1;
      v36[0] = v18;
      v24 = v21;
      v36[1] = v19;
      if ( v21 + 1 > v20 )
      {
        if ( v22 > (unsigned __int64)v36 )
        {
          v31 = v19;
          v32 = v18;
LABEL_11:
          sub_2AF6270(v33, v23, v21, v22, v19, v9);
          v21 = *(unsigned int *)(a1 + 184);
          v22 = *(_QWORD *)(a1 + 176);
          v18 = v32;
          v19 = v31;
          v24 = *(_DWORD *)(a1 + 184);
          goto LABEL_3;
        }
        v31 = v19;
        v32 = v18;
        v21 = v22 + 48 * v21;
        if ( (unsigned __int64)v36 >= v21 )
          goto LABEL_11;
        v30 = (char *)v36 - v22;
        sub_2AF6270(v33, v23, v21, v22, v19, v9);
        v22 = *(_QWORD *)(a1 + 176);
        v21 = *(unsigned int *)(a1 + 184);
        v19 = v31;
        v18 = v32;
        v16 = (const __m128i *)&v30[v22];
        v24 = *(_DWORD *)(a1 + 184);
      }
LABEL_3:
      v25 = (__m128i *)(v22 + 48 * v21);
      if ( v25 )
      {
        v26 = _mm_loadu_si128(v16 + 1);
        *v25 = _mm_loadu_si128(v16);
        v25[1] = v26;
        v25[2].m128i_i32[2] = v16[2].m128i_i32[2];
        v27 = v16[2].m128i_i8[12];
        v25[2].m128i_i64[0] = (__int64)&unk_4A23530;
        v25[2].m128i_i8[12] = v27;
        v24 = *(_DWORD *)(a1 + 184);
      }
      v28 = *(_QWORD *)(a1 + 168);
      v13 += 40;
      *(_DWORD *)(a1 + 184) = v24 + 1;
      sub_C52F90(v28, v18, v19);
    }
    while ( v35 != v13 );
  }
  return sub_C53130(a1);
}
