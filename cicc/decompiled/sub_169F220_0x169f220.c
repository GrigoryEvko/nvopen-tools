// Function: sub_169F220
// Address: 0x169f220
//
__int64 __fastcall sub_169F220(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 *v4; // rdi
  void *v5; // r13
  float v6; // xmm0_4
  __int64 *v7; // rdi
  float v8; // xmm0_4
  __int64 *v9; // rdi
  void *v10; // r13
  float v11; // xmm0_4
  __int64 *v12; // rdi
  float v13; // xmm0_4
  __int16 *v14; // r15
  __int64 *v16; // rdi
  void *v17; // r13
  float v18; // xmm0_4
  __int64 *v19; // rdi
  float v20; // xmm0_4
  __int64 *v21; // rdi
  void *v22; // r13
  float v23; // xmm0_4
  __int64 *v24; // rdi
  float v25; // [rsp+Ch] [rbp-84h]
  float v26; // [rsp+Ch] [rbp-84h]
  float v27; // [rsp+Ch] [rbp-84h]
  float v28; // [rsp+Ch] [rbp-84h]
  float v29; // [rsp+Ch] [rbp-84h]
  char v30; // [rsp+1Fh] [rbp-71h] BYREF
  __int64 v31[5]; // [rsp+20h] [rbp-70h] BYREF
  void *v32[9]; // [rsp+48h] [rbp-48h] BYREF

  if ( a3 == 2 )
  {
    v21 = (__int64 *)(a2 + 8);
    v22 = sub_16982C0();
    if ( *(void **)(a2 + 8) == v22 )
      v21 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
    v23 = sub_169D890(v21);
    v24 = (__int64 *)(a1 + 8);
    if ( *(void **)(a1 + 8) == v22 )
      v24 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
    v29 = v23;
    v8 = sub_169D890(v24);
    sub_1C40ED0(&v30, 1, 1, v8, v29);
    goto LABEL_15;
  }
  if ( a3 <= 2 )
  {
    if ( !a3 )
    {
      v16 = (__int64 *)(a2 + 8);
      v17 = sub_16982C0();
      if ( *(void **)(a2 + 8) == v17 )
        v16 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
      v18 = sub_169D890(v16);
      v19 = (__int64 *)(a1 + 8);
      if ( *(void **)(a1 + 8) == v17 )
        v19 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
      v28 = v18;
      v20 = sub_169D890(v19);
      v8 = sub_1C40EA0(&v30, 1, 1, v20, v28);
      if ( !(unsigned int)sub_1C40EE0(&v30) )
      {
        v27 = v8;
        goto LABEL_16;
      }
      goto LABEL_22;
    }
    v4 = (__int64 *)(a2 + 8);
    v5 = sub_16982C0();
    if ( *(void **)(a2 + 8) == v5 )
      v4 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
    v6 = sub_169D890(v4);
    v7 = (__int64 *)(a1 + 8);
    if ( *(void **)(a1 + 8) == v5 )
      v7 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
    v25 = v6;
    v8 = sub_169D890(v7);
    sub_1C40EC0(&v30, 1, 1, v8, v25);
LABEL_15:
    v27 = v8;
    if ( !(unsigned int)sub_1C40EE0(&v30) )
    {
LABEL_16:
      v14 = (__int16 *)sub_1698270();
      sub_169D3B0((__int64)v31, (__m128i)LODWORD(v27));
      sub_169E320(v32, v31, v14);
      sub_1698460((__int64)v31);
      sub_169ED90((void **)(a1 + 8), v32);
      sub_127D120(v32);
      return 0;
    }
LABEL_22:
    sub_169CB40(a1, 0, 0, 0, v8);
    return 1;
  }
  if ( a3 == 3 )
  {
    v9 = (__int64 *)(a2 + 8);
    v10 = sub_16982C0();
    if ( *(void **)(a2 + 8) == v10 )
      v9 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
    v11 = sub_169D890(v9);
    v12 = (__int64 *)(a1 + 8);
    if ( *(void **)(a1 + 8) == v10 )
      v12 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
    v26 = v11;
    v13 = sub_169D890(v12);
    v8 = sub_1C40EB0(&v30, 1, 1, v13, v26);
    goto LABEL_15;
  }
  return 1;
}
