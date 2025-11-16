// Function: sub_169F510
// Address: 0x169f510
//
__int64 __fastcall sub_169F510(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 *v6; // rdi
  void *v7; // r14
  float v8; // xmm0_4
  __int64 *v9; // rdi
  float v10; // xmm0_4
  __int64 *v11; // rdi
  float v12; // xmm0_4
  __int64 *v13; // rdi
  void *v14; // r14
  float v15; // xmm0_4
  __int64 *v16; // rdi
  float v17; // xmm0_4
  __int64 *v18; // rdi
  float v19; // xmm0_4
  __int16 *v20; // r15
  __int64 *v22; // rdi
  void *v23; // r14
  float v24; // xmm0_4
  __int64 *v25; // rdi
  float v26; // xmm0_4
  __int64 *v27; // rdi
  float v28; // xmm0_4
  __int64 *v29; // rdi
  void *v30; // r14
  float v31; // xmm0_4
  __int64 *v32; // rdi
  float v33; // xmm0_4
  __int64 *v34; // rdi
  float v35; // [rsp+8h] [rbp-88h]
  float v36; // [rsp+8h] [rbp-88h]
  float v37; // [rsp+8h] [rbp-88h]
  float v38; // [rsp+8h] [rbp-88h]
  float v39; // [rsp+Ch] [rbp-84h]
  float v40; // [rsp+Ch] [rbp-84h]
  float v41; // [rsp+Ch] [rbp-84h]
  float v42; // [rsp+Ch] [rbp-84h]
  float v43; // [rsp+Ch] [rbp-84h]
  float v44; // [rsp+Ch] [rbp-84h]
  float v45; // [rsp+Ch] [rbp-84h]
  float v46; // [rsp+Ch] [rbp-84h]
  float v47; // [rsp+Ch] [rbp-84h]
  char v48; // [rsp+1Fh] [rbp-71h] BYREF
  __int64 v49[5]; // [rsp+20h] [rbp-70h] BYREF
  void *v50[9]; // [rsp+48h] [rbp-48h] BYREF

  if ( a4 == 2 )
  {
    v29 = (__int64 *)(a3 + 8);
    v30 = sub_16982C0();
    if ( *(void **)(a3 + 8) == v30 )
      v29 = (__int64 *)(*(_QWORD *)(a3 + 16) + 8LL);
    v31 = sub_169D890(v29);
    v32 = (__int64 *)(a2 + 8);
    if ( *(void **)(a2 + 8) == v30 )
      v32 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
    v46 = v31;
    v33 = sub_169D890(v32);
    v34 = (__int64 *)(a1 + 8);
    if ( *(void **)(a1 + 8) == v30 )
      v34 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
    v38 = v46;
    v47 = v33;
    v12 = sub_169D890(v34);
    sub_1C40E90(&v48, 1, 1, v12, v47, v38);
    goto LABEL_19;
  }
  if ( a4 <= 2 )
  {
    if ( !a4 )
    {
      v22 = (__int64 *)(a3 + 8);
      v23 = sub_16982C0();
      if ( *(void **)(a3 + 8) == v23 )
        v22 = (__int64 *)(*(_QWORD *)(a3 + 16) + 8LL);
      v24 = sub_169D890(v22);
      v25 = (__int64 *)(a2 + 8);
      if ( *(void **)(a2 + 8) == v23 )
        v25 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
      v44 = v24;
      v26 = sub_169D890(v25);
      v27 = (__int64 *)(a1 + 8);
      if ( *(void **)(a1 + 8) == v23 )
        v27 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
      v37 = v44;
      v45 = v26;
      v28 = sub_169D890(v27);
      v12 = sub_1C40E60(&v48, 1, 1, v28, v45, v37);
      if ( !(unsigned int)sub_1C40EE0(&v48) )
      {
        v43 = v12;
        goto LABEL_20;
      }
      goto LABEL_28;
    }
    v6 = (__int64 *)(a3 + 8);
    v7 = sub_16982C0();
    if ( *(void **)(a3 + 8) == v7 )
      v6 = (__int64 *)(*(_QWORD *)(a3 + 16) + 8LL);
    v8 = sub_169D890(v6);
    v9 = (__int64 *)(a2 + 8);
    if ( *(void **)(a2 + 8) == v7 )
      v9 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
    v39 = v8;
    v10 = sub_169D890(v9);
    v11 = (__int64 *)(a1 + 8);
    if ( *(void **)(a1 + 8) == v7 )
      v11 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
    v35 = v39;
    v40 = v10;
    v12 = sub_169D890(v11);
    sub_1C40E80(&v48, 1, 1, v12, v40, v35);
LABEL_19:
    v43 = v12;
    if ( !(unsigned int)sub_1C40EE0(&v48) )
    {
LABEL_20:
      v20 = (__int16 *)sub_1698270();
      sub_169D3B0((__int64)v49, (__m128i)LODWORD(v43));
      sub_169E320(v50, v49, v20);
      sub_1698460((__int64)v49);
      sub_169ED90((void **)(a1 + 8), v50);
      sub_127D120(v50);
      return 0;
    }
LABEL_28:
    sub_169CB40(a1, 0, 0, 0, v12);
    return 1;
  }
  if ( a4 == 3 )
  {
    v13 = (__int64 *)(a3 + 8);
    v14 = sub_16982C0();
    if ( *(void **)(a3 + 8) == v14 )
      v13 = (__int64 *)(*(_QWORD *)(a3 + 16) + 8LL);
    v15 = sub_169D890(v13);
    v16 = (__int64 *)(a2 + 8);
    if ( *(void **)(a2 + 8) == v14 )
      v16 = (__int64 *)(*(_QWORD *)(a2 + 16) + 8LL);
    v41 = v15;
    v17 = sub_169D890(v16);
    v18 = (__int64 *)(a1 + 8);
    if ( *(void **)(a1 + 8) == v14 )
      v18 = (__int64 *)(*(_QWORD *)(a1 + 16) + 8LL);
    v36 = v41;
    v42 = v17;
    v19 = sub_169D890(v18);
    v12 = sub_1C40E70(&v48, 1, 1, v19, v42, v36);
    goto LABEL_19;
  }
  return 1;
}
