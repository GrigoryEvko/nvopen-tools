// Function: sub_2123AD0
// Address: 0x2123ad0
//
__int64 __fastcall sub_2123AD0(__m128i **a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int64 *v7; // rdx
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  char v11; // cl
  unsigned __int8 *v12; // rax
  char v13; // r14
  const void **v14; // r13
  unsigned int v15; // r15d
  __int64 v16; // rdx
  char v17; // di
  __int64 *v18; // r14
  int v19; // eax
  __int64 v20; // rsi
  __m128i *v21; // r10
  int v22; // ecx
  __int64 v23; // r11
  __int64 v24; // rsi
  __int64 v25; // r12
  __int64 v27; // rsi
  __int64 *v28; // r12
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-80h]
  int v31; // [rsp+0h] [rbp-80h]
  char v32; // [rsp+8h] [rbp-78h]
  __m128i *v33; // [rsp+8h] [rbp-78h]
  __int128 v34; // [rsp+10h] [rbp-70h] BYREF
  __int64 v35; // [rsp+20h] [rbp-60h] BYREF
  int v36; // [rsp+28h] [rbp-58h]
  __int64 v37; // [rsp+30h] [rbp-50h] BYREF
  __int64 v38; // [rsp+38h] [rbp-48h]

  v7 = *(unsigned __int64 **)(a2 + 32);
  v8 = *v7;
  v9 = v7[1];
  v10 = *(_QWORD *)(v8 + 40) + 16LL * (unsigned int)v9;
  v11 = *(_BYTE *)v10;
  v30 = *(_QWORD *)(v10 + 8);
  v12 = *(unsigned __int8 **)(a2 + 40);
  v13 = v11;
  v32 = v11;
  v14 = (const void **)*((_QWORD *)v12 + 1);
  v15 = *v12;
  *(_QWORD *)&v34 = sub_2120330((__int64)a1, v8, v9);
  *((_QWORD *)&v34 + 1) = v16;
  if ( v32 == 8 )
  {
    v27 = *(_QWORD *)(a2 + 72);
    v28 = (__int64 *)a1[1];
    v18 = &v37;
    v37 = v27;
    if ( v27 )
      sub_1623A60((__int64)&v37, v27, 2);
    LODWORD(v38) = *(_DWORD *)(a2 + 64);
    v29 = sub_1D309E0(
            v28,
            160,
            (__int64)&v37,
            v15,
            v14,
            0,
            *(double *)a3.m128i_i64,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            v34);
    v24 = v37;
    v25 = v29;
    if ( v37 )
      goto LABEL_5;
  }
  else
  {
    v17 = v13;
    v18 = &v35;
    v19 = sub_1F3FE80(v17, v30, v15);
    v20 = *(_QWORD *)(a2 + 72);
    v21 = *a1;
    v22 = v19;
    v35 = v20;
    if ( v20 )
    {
      v18 = &v35;
      v33 = v21;
      v31 = v19;
      sub_1623A60((__int64)&v35, v20, 2);
      v22 = v31;
      v21 = v33;
    }
    v23 = (__int64)a1[1];
    v36 = *(_DWORD *)(a2 + 64);
    sub_20BE530((__int64)&v37, v21, v23, v22, v15, (__int64)v14, a3, a4, a5, (__int64)&v34, 1u, 0, (__int64)&v35, 0, 1);
    v24 = v35;
    v25 = v37;
    if ( v35 )
LABEL_5:
      sub_161E7C0((__int64)v18, v24);
  }
  return v25;
}
