// Function: sub_32972C0
// Address: 0x32972c0
//
__int64 __fastcall sub_32972C0(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 v7; // r13
  unsigned int v8; // ecx
  __m128i v9; // xmm1
  __int64 v10; // r14
  __int64 v11; // rax
  __int16 v12; // r11
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rdi
  __m128i v16; // xmm3
  __m128i v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 v20; // r8
  bool v21; // al
  char v22; // al
  __int64 v23; // rdi
  __int64 v24; // r8
  int v25; // eax
  int v26; // ecx
  __int64 v28; // [rsp-8h] [rbp-B8h]
  __int64 v29; // [rsp+0h] [rbp-B0h]
  __int16 v30; // [rsp+8h] [rbp-A8h]
  __int16 v31; // [rsp+8h] [rbp-A8h]
  __int64 v32; // [rsp+8h] [rbp-A8h]
  __int64 v33; // [rsp+8h] [rbp-A8h]
  __int64 v34; // [rsp+8h] [rbp-A8h]
  __m128i v35; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v36; // [rsp+20h] [rbp-90h]
  int v37; // [rsp+24h] [rbp-8Ch]
  __int64 v38; // [rsp+28h] [rbp-88h]
  __m128i v39; // [rsp+30h] [rbp-80h] BYREF
  int v40; // [rsp+40h] [rbp-70h] BYREF
  __int64 v41; // [rsp+48h] [rbp-68h]
  __int64 v42; // [rsp+50h] [rbp-60h] BYREF
  int v43; // [rsp+58h] [rbp-58h]
  _OWORD v44[5]; // [rsp+60h] [rbp-50h] BYREF

  v37 = *(_DWORD *)(a2 + 24);
  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128((const __m128i *)v4);
  v7 = *v4;
  v8 = *((_DWORD *)v4 + 2);
  v9 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v10 = v4[5];
  LODWORD(v4) = *((_DWORD *)v4 + 12);
  v35 = v6;
  v39 = v9;
  LODWORD(v38) = (_DWORD)v4;
  v29 = v8;
  v11 = *(_QWORD *)(v7 + 48) + 16LL * v8;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v42 = v5;
  LOWORD(v40) = v12;
  v41 = v13;
  if ( v5 )
  {
    v36 = v8;
    v30 = v12;
    sub_B96E90((__int64)&v42, v5, 1);
    v8 = v36;
    v12 = v30;
  }
  v14 = *(_DWORD *)(v7 + 24) == 51;
  v15 = *a1;
  v43 = *(_DWORD *)(a2 + 72);
  if ( v14 || *(_DWORD *)(v10 + 24) == 51 || (_DWORD)v38 == v8 && v10 == v7 )
  {
    v17.m128i_i64[0] = sub_3400BD0(v15, 0, (unsigned int)&v42, v40, v41, 0, 0);
    goto LABEL_13;
  }
  v16 = _mm_load_si128(&v39);
  v31 = v12;
  v44[0] = _mm_load_si128(&v35);
  v44[1] = v16;
  v17.m128i_i64[0] = sub_3402EA0(v15, v37, (unsigned int)&v42, v40, v41, 0, (__int64)v44, 2);
  v19 = v17.m128i_i64[1];
  v20 = v28;
  if ( v17.m128i_i64[0] )
    goto LABEL_13;
  if ( v31 )
  {
    if ( (unsigned __int16)(v31 - 17) > 0xD3u )
      goto LABEL_10;
  }
  else
  {
    v33 = v17.m128i_i64[1];
    v21 = sub_30070B0((__int64)&v40);
    v19 = v33;
    if ( !v21 )
      goto LABEL_10;
  }
  v34 = v19;
  v17.m128i_i64[0] = sub_3295970(a1, a2, (__int64)&v42, v18, v20);
  if ( v17.m128i_i64[0] )
    goto LABEL_13;
  v22 = sub_33D1AE0(v10, 0);
  v19 = v34;
  if ( v22 )
    goto LABEL_11;
LABEL_10:
  v32 = v19;
  if ( (unsigned __int8)sub_33CF170(v39.m128i_i64[0], v39.m128i_i64[1]) )
  {
LABEL_11:
    v17 = v35;
    goto LABEL_13;
  }
  v23 = *a1;
  v24 = (unsigned int)v38;
  v38 = v32;
  if ( v37 == 84 )
    v25 = sub_33DF620(v23, v7, v29, v10, v24, v32);
  else
    v25 = sub_33DD890(v23, v7, v29, v10, v24, v32);
  v26 = v25;
  v17.m128i_i64[0] = 0;
  v17.m128i_i64[1] = v38 & 0xFFFFFFFF00000000LL;
  if ( !v26 )
    v17.m128i_i64[0] = sub_3406EB0(*a1, 57, (unsigned int)&v42, v40, v41, v38, *(_OWORD *)&v35, *(_OWORD *)&v39);
LABEL_13:
  if ( v42 )
  {
    v38 = v17.m128i_i64[1];
    v39.m128i_i64[0] = v17.m128i_i64[0];
    sub_B91220((__int64)&v42, v42);
    v17.m128i_i64[0] = v39.m128i_i64[0];
  }
  return v17.m128i_i64[0];
}
