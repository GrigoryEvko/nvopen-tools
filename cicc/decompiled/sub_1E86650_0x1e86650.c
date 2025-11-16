// Function: sub_1E86650
// Address: 0x1e86650
//
__int64 __fastcall sub_1E86650(__int64 a1)
{
  __int8 *v1; // r13
  size_t v2; // r12
  __m128i *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  __m128i *v9; // rax
  size_t v10; // rax
  __int64 v11; // rax
  __m128i *v13; // rdi
  __int64 v14; // [rsp+18h] [rbp-48h] BYREF
  __m128i *v15; // [rsp+20h] [rbp-40h] BYREF
  size_t v16; // [rsp+28h] [rbp-38h]
  __m128i v17[3]; // [rsp+30h] [rbp-30h] BYREF

  v1 = *(__int8 **)a1;
  v2 = *(_QWORD *)(a1 + 8);
  v15 = v17;
  if ( &v1[v2] && !v1 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v14 = v2;
  if ( v2 > 0xF )
  {
    v15 = (__m128i *)sub_22409D0(&v15, &v14, 0);
    v13 = v15;
    v17[0].m128i_i64[0] = v14;
  }
  else
  {
    if ( v2 == 1 )
    {
      v17[0].m128i_i8[0] = *v1;
      v3 = v17;
      goto LABEL_6;
    }
    if ( !v2 )
    {
      v3 = v17;
      goto LABEL_6;
    }
    v13 = v17;
  }
  memcpy(v13, v1, v2);
  v2 = v14;
  v3 = v15;
LABEL_6:
  v16 = v2;
  v3->m128i_i8[v2] = 0;
  v4 = sub_22077B0(264);
  v5 = v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)(v4 + 16) = &unk_4FC8214;
    *(_QWORD *)(v4 + 80) = v4 + 64;
    *(_QWORD *)(v4 + 88) = v4 + 64;
    *(_QWORD *)(v4 + 128) = v4 + 112;
    *(_QWORD *)(v4 + 136) = v4 + 112;
    *(_DWORD *)(v4 + 24) = 3;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    *(_DWORD *)(v4 + 64) = 0;
    *(_QWORD *)(v4 + 72) = 0;
    *(_QWORD *)(v4 + 96) = 0;
    *(_DWORD *)(v4 + 112) = 0;
    *(_QWORD *)(v4 + 120) = 0;
    *(_QWORD *)(v4 + 144) = 0;
    *(_BYTE *)(v4 + 152) = 0;
    *(_QWORD *)v4 = &unk_49FB790;
    *(_QWORD *)(v4 + 160) = 0;
    *(_QWORD *)(v4 + 168) = 0;
    *(_DWORD *)(v4 + 176) = 8;
    v6 = (_QWORD *)malloc(8u);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = 0;
    }
    *(_QWORD *)(v5 + 160) = v6;
    *(_QWORD *)(v5 + 168) = 1;
    *v6 = 0;
    *(_QWORD *)(v5 + 184) = 0;
    *(_QWORD *)(v5 + 192) = 0;
    *(_DWORD *)(v5 + 200) = 8;
    v7 = (_QWORD *)malloc(8u);
    if ( !v7 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v7 = 0;
    }
    *(_QWORD *)(v5 + 184) = v7;
    *(_QWORD *)(v5 + 192) = 1;
    *v7 = 0;
    *(_QWORD *)(v5 + 208) = 0;
    *(_QWORD *)(v5 + 216) = 0;
    *(_DWORD *)(v5 + 224) = 8;
    v8 = (_QWORD *)malloc(8u);
    if ( !v8 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v8 = 0;
    }
    *(_QWORD *)(v5 + 208) = v8;
    *v8 = 0;
    *(_QWORD *)v5 = off_49FCE40;
    *(_QWORD *)(v5 + 232) = v5 + 248;
    v9 = v15;
    *(_QWORD *)(v5 + 216) = 1;
    if ( v9 == v17 )
    {
      *(__m128i *)(v5 + 248) = _mm_load_si128(v17);
    }
    else
    {
      *(_QWORD *)(v5 + 232) = v9;
      *(_QWORD *)(v5 + 248) = v17[0].m128i_i64[0];
    }
    v10 = v16;
    v15 = v17;
    v16 = 0;
    *(_QWORD *)(v5 + 240) = v10;
    v17[0].m128i_i8[0] = 0;
    v11 = sub_163A1D0();
    sub_1E862A0(v11);
  }
  if ( v15 != v17 )
    j_j___libc_free_0(v15, v17[0].m128i_i64[0] + 1);
  return v5;
}
