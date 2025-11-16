// Function: sub_38C2930
// Address: 0x38c2930
//
__int64 __fastcall sub_38C2930(__int64 a1, __int64 a2, unsigned __int8 a3, _BYTE *a4, int a5)
{
  const char *v5; // r15
  __int64 v8; // rax
  __m128i v9; // xmm0
  unsigned __int64 v10; // rax
  char v11; // dl
  _QWORD *v12; // r15
  __int64 v13; // r8
  _QWORD *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r14
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  void *v25; // [rsp+0h] [rbp-F0h]
  unsigned __int64 v27; // [rsp+10h] [rbp-E0h]
  __int64 v28; // [rsp+18h] [rbp-D8h]
  char v29; // [rsp+18h] [rbp-D8h]
  __int64 v30; // [rsp+18h] [rbp-D8h]
  _BYTE *v31[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+30h] [rbp-C0h] BYREF
  __m128i *v33; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+48h] [rbp-A8h]
  __m128i v35; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v36; // [rsp+60h] [rbp-90h] BYREF
  int v37; // [rsp+70h] [rbp-80h]
  __m128i v38; // [rsp+80h] [rbp-70h] BYREF
  _OWORD v39[2]; // [rsp+90h] [rbp-60h] BYREF
  int v40; // [rsp+B0h] [rbp-40h]
  __int64 v41; // [rsp+B8h] [rbp-38h]

  v5 = byte_3F871B3;
  v28 = 0;
  if ( a4 )
  {
    v5 = 0;
    if ( (*a4 & 4) != 0 )
    {
      v15 = (_QWORD *)*((_QWORD *)a4 - 1);
      v16 = *v15;
      v5 = (const char *)(v15 + 2);
      v28 = v16;
    }
  }
  sub_16E2FC0((__int64 *)v31, a2);
  if ( v31[0] )
  {
    v33 = &v35;
    sub_38BB9D0((__int64 *)&v33, v31[0], (__int64)&v31[0][(unsigned __int64)v31[1]]);
    v36.m128i_i64[0] = (__int64)v5;
    v37 = a5;
    v8 = v34;
    v36.m128i_i64[1] = v28;
    v38.m128i_i64[0] = (__int64)v39;
    if ( v33 != &v35 )
    {
      v38.m128i_i64[0] = (__int64)v33;
      *(_QWORD *)&v39[0] = v35.m128i_i64[0];
      goto LABEL_7;
    }
  }
  else
  {
    v37 = a5;
    v35.m128i_i8[0] = 0;
    v36.m128i_i64[0] = (__int64)v5;
    v38.m128i_i64[0] = (__int64)v39;
    v36.m128i_i64[1] = v28;
    v8 = 0;
  }
  v39[0] = _mm_load_si128(&v35);
LABEL_7:
  v9 = _mm_load_si128(&v36);
  v38.m128i_i64[1] = v8;
  v33 = &v35;
  v34 = 0;
  v35.m128i_i8[0] = 0;
  v40 = a5;
  v41 = 0;
  v39[1] = v9;
  v10 = sub_38C2480((_QWORD *)(a1 + 1296), &v38);
  v29 = v11;
  v12 = (_QWORD *)v10;
  if ( (_OWORD *)v38.m128i_i64[0] != v39 )
    j_j___libc_free_0(v38.m128i_u64[0]);
  if ( v33 != &v35 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( (__int64 *)v31[0] != &v32 )
    j_j___libc_free_0((unsigned __int64)v31[0]);
  if ( !v29 )
    return v12[11];
  v25 = (void *)v12[4];
  v27 = v12[5];
  v17 = sub_38BEE30(a1, v25, v27, 0, 0, v27);
  *(_DWORD *)(v17 + 32) = 3;
  v18 = (_QWORD *)v17;
  v30 = sub_145CBF0((__int64 *)(a1 + 464), 200, 8);
  sub_38D76F0(v30, 3, a3, v18);
  *(_DWORD *)(v30 + 168) = a5;
  *(_QWORD *)v30 = &unk_4A3E5E8;
  *(_QWORD *)(v30 + 152) = v25;
  *(_QWORD *)(v30 + 160) = v27;
  *(_QWORD *)(v30 + 176) = a4;
  *(_QWORD *)(v30 + 184) = 0;
  *(_DWORD *)(v30 + 192) = 0;
  v12[11] = v30;
  v19 = sub_22077B0(0xE0u);
  v13 = v30;
  v20 = v19;
  if ( v19 )
  {
    v21 = v19;
    sub_38CF760(v19, 1, 0, 0);
    *(_QWORD *)(v20 + 56) = 0;
    v13 = v30;
    *(_WORD *)(v20 + 48) = 0;
    *(_QWORD *)(v20 + 64) = v20 + 80;
    *(_QWORD *)(v20 + 72) = 0x2000000000LL;
    *(_QWORD *)(v20 + 112) = v20 + 128;
    *(_QWORD *)(v20 + 120) = 0x400000000LL;
  }
  else
  {
    v21 = 0;
  }
  v22 = *(__int64 **)(v13 + 104);
  v23 = *v22;
  v24 = *(_QWORD *)v20 & 7LL;
  *(_QWORD *)(v20 + 8) = v22;
  v23 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v20 = v23 | v24;
  *(_QWORD *)(v23 + 8) = v21;
  *v22 = *v22 & 7 | v21;
  *(_QWORD *)(v20 + 24) = v13;
  *v18 = *v18 & 7LL | v20;
  return v13;
}
