// Function: sub_2D46B10
// Address: 0x2d46b10
//
__int64 __fastcall sub_2D46B10(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 v6; // rax
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __m128i v9; // xmm4
  __m128i v10; // xmm5
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  void (__fastcall *v13)(__int64, __m128i *, __int64); // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  int v19; // ecx
  _QWORD *v20; // rdx
  __int64 result; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __m128i v24; // [rsp+10h] [rbp-E0h] BYREF
  void (__fastcall *v25)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-D0h]
  void (__fastcall *v26)(__int64 *, unsigned __int8 **); // [rsp+28h] [rbp-C8h]
  void *v27; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v28; // [rsp+38h] [rbp-B8h] BYREF
  void (__fastcall *v29)(__int64, __m128i *, __int64); // [rsp+48h] [rbp-A8h]
  void (__fastcall *v30)(__int64 *, unsigned __int8 **); // [rsp+50h] [rbp-A0h]
  void *v31; // [rsp+60h] [rbp-90h] BYREF
  void *v32; // [rsp+68h] [rbp-88h]
  unsigned __int64 v33; // [rsp+70h] [rbp-80h]
  __m128i v34; // [rsp+78h] [rbp-78h] BYREF
  __m128i v35; // [rsp+88h] [rbp-68h] BYREF
  __m128i v36; // [rsp+98h] [rbp-58h] BYREF
  __m128i v37; // [rsp+A8h] [rbp-48h] BYREF
  __int64 v38; // [rsp+B8h] [rbp-38h]

  v33 = a3;
  v24.m128i_i64[0] = a1;
  v4 = _mm_load_si128(&v24);
  v34 = (__m128i)a3;
  v5 = _mm_loadu_si128(&v28);
  v29 = (void (__fastcall *)(__int64, __m128i *, __int64))sub_2D42C70;
  v24 = v5;
  v26 = v30;
  v30 = sub_2D42BD0;
  v28 = v4;
  v27 = &unk_49DA0D8;
  v31 = &unk_49E5698;
  v25 = 0;
  v32 = &unk_49D94D0;
  v35 = 0u;
  v36 = 0u;
  v37 = 0u;
  LOWORD(v38) = 257;
  v6 = sub_BD5C60(a2);
  v7 = _mm_loadu_si128(&v34);
  *(_QWORD *)(a1 + 72) = v6;
  v8 = _mm_loadu_si128(&v35);
  v9 = _mm_loadu_si128(&v36);
  v10 = _mm_loadu_si128(&v37);
  *(_QWORD *)(a1 + 80) = a1 + 128;
  *(_QWORD *)(a1 + 88) = a1 + 224;
  v11 = v33;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  *(_QWORD *)(a1 + 144) = v11;
  v12 = v38;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 216) = v12;
  v13 = v29;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_WORD *)(a1 + 108) = 512;
  *(_BYTE *)(a1 + 110) = 7;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_WORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 128) = &unk_49E5698;
  *(_QWORD *)(a1 + 136) = &unk_49D94D0;
  *(_QWORD *)(a1 + 224) = &unk_49DA0D8;
  *(_QWORD *)(a1 + 248) = 0;
  *(__m128i *)(a1 + 152) = v7;
  *(__m128i *)(a1 + 168) = v8;
  *(__m128i *)(a1 + 184) = v9;
  *(__m128i *)(a1 + 200) = v10;
  if ( v13 )
  {
    v13(a1 + 232, &v28, 2);
    *(_QWORD *)(a1 + 256) = v30;
    *(_QWORD *)(a1 + 248) = v29;
  }
  v31 = &unk_49E5698;
  v32 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  sub_B32BF0(&v27);
  if ( v25 )
    v25(&v24, &v24, 3);
  *(_QWORD *)(a1 + 264) = 0;
  sub_D5F1F0(a1, a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && (v16 = sub_B91C10(a2, 37)) != 0 )
  {
    v17 = *(unsigned int *)(a1 + 8);
    v18 = *(_QWORD **)a1;
    v19 = *(_DWORD *)(a1 + 8);
    v20 = (_QWORD *)(*(_QWORD *)a1 + 16 * v17);
    if ( *(_QWORD **)a1 == v20 )
    {
LABEL_17:
      v22 = *(unsigned int *)(a1 + 12);
      if ( v17 >= v22 )
      {
        v23 = v17 + 1;
        if ( v22 < v23 )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v23, 0x10u, v14, v15);
          v20 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v20 = 37;
        v20[1] = v16;
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v20 )
        {
          *(_DWORD *)v20 = 37;
          v20[1] = v16;
          v19 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v19 + 1;
      }
    }
    else
    {
      while ( *(_DWORD *)v18 != 37 )
      {
        v18 += 2;
        if ( v20 == v18 )
          goto LABEL_17;
      }
      v18[1] = v16;
    }
  }
  else
  {
    sub_93FB40(a1, 37);
  }
  v31 = *(void **)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 72LL) + 120LL);
  if ( (unsigned __int8)sub_A73ED0(&v31, 72) )
    *(_BYTE *)(a1 + 108) = 1;
  result = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    result = sub_B91C10(a2, 40);
  *(_QWORD *)(a1 + 264) = result;
  return result;
}
