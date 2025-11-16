// Function: sub_37A0050
// Address: 0x37a0050
//
unsigned __int8 *__fastcall sub_37A0050(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  unsigned __int8 *v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // r14
  int v13; // r9d
  __int64 v14; // rsi
  unsigned __int8 *v15; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rsi
  const __m128i *v21; // rax
  _QWORD *v22; // r9
  __m128i v23; // xmm0
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int128 v26; // [rsp-10h] [rbp-C0h]
  int v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  _QWORD *v29; // [rsp+8h] [rbp-A8h]
  unsigned int v30; // [rsp+30h] [rbp-80h] BYREF
  __int64 v31; // [rsp+38h] [rbp-78h]
  __int64 v32; // [rsp+40h] [rbp-70h] BYREF
  int v33; // [rsp+48h] [rbp-68h]
  __int64 v34; // [rsp+50h] [rbp-60h] BYREF
  __int64 v35; // [rsp+58h] [rbp-58h]
  __int64 v36; // [rsp+60h] [rbp-50h]
  unsigned __int64 v37; // [rsp+68h] [rbp-48h]
  __m128i v38; // [rsp+70h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v34, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v30) = v35;
    v31 = v36;
  }
  else
  {
    v30 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v31 = v25;
  }
  v8 = (unsigned __int8 *)sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = v9;
  if ( *(_DWORD *)(a2 + 64) == 1 )
  {
    v11 = *(_QWORD *)(a2 + 80);
    v12 = a1[1];
    v13 = *(_DWORD *)(a2 + 28);
    v34 = v11;
    if ( v11 )
    {
      v27 = v13;
      sub_B96E90((__int64)&v34, v11, 1);
      v13 = v27;
    }
    v14 = *(unsigned int *)(a2 + 24);
    LODWORD(v35) = *(_DWORD *)(a2 + 72);
    v15 = sub_33FA050(v12, v14, (__int64)&v34, v30, v31, v13, v8, v10);
    if ( v34 )
      sub_B91220((__int64)&v34, v34);
  }
  else
  {
    if ( !(_WORD)v30 )
      sub_3007240((__int64)&v30);
    v17 = *(_QWORD *)(a2 + 40);
    v28 = *(_QWORD *)(v17 + 48);
    v18 = sub_379AB60((__int64)a1, *(_QWORD *)(v17 + 40), v28);
    v34 = (__int64)v8;
    v20 = *(_QWORD *)(a2 + 80);
    v36 = v18;
    v21 = *(const __m128i **)(a2 + 40);
    v35 = v10;
    v22 = (_QWORD *)a1[1];
    v37 = v19 | v28 & 0xFFFFFFFF00000000LL;
    v23 = _mm_loadu_si128(v21 + 5);
    v32 = v20;
    v38 = v23;
    if ( v20 )
    {
      v29 = v22;
      sub_B96E90((__int64)&v32, v20, 1);
      v22 = v29;
    }
    *((_QWORD *)&v26 + 1) = 3;
    v24 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)&v26 = &v34;
    v33 = *(_DWORD *)(a2 + 72);
    v15 = sub_33FC220(v22, v24, (__int64)&v32, v30, v31, (__int64)v22, v26);
    if ( v32 )
      sub_B91220((__int64)&v32, v32);
  }
  return v15;
}
