// Function: sub_38159C0
// Address: 0x38159c0
//
unsigned __int8 *__fastcall sub_38159C0(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r11
  __int64 v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int16 *v9; // rax
  __int64 v10; // r13
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 (__fastcall *v14)(__int64, __int64, __int64, __int64, __int64); // r14
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rdx
  unsigned int v18; // r10d
  unsigned int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // r11
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __int64 v24; // rsi
  __int64 *v25; // rdi
  __int64 v26; // r15
  __int64 v27; // rax
  unsigned int *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r9
  unsigned __int8 *v31; // r14
  __int64 v32; // rdx
  unsigned __int8 *v33; // r14
  unsigned __int8 *v35; // rax
  int v36; // edx
  __int64 v37; // rdx
  __int128 v38; // [rsp-20h] [rbp-F0h]
  __int128 v39; // [rsp-10h] [rbp-E0h]
  unsigned int v40; // [rsp+0h] [rbp-D0h]
  unsigned int v41; // [rsp+0h] [rbp-D0h]
  __int64 v42; // [rsp+8h] [rbp-C8h]
  __int64 v43; // [rsp+8h] [rbp-C8h]
  __int64 v44; // [rsp+10h] [rbp-C0h]
  unsigned int v45; // [rsp+10h] [rbp-C0h]
  __int64 v46; // [rsp+20h] [rbp-B0h]
  unsigned int v47; // [rsp+28h] [rbp-A8h]
  unsigned __int16 v48; // [rsp+36h] [rbp-9Ah]
  __int64 v49; // [rsp+38h] [rbp-98h]
  __int64 v50; // [rsp+40h] [rbp-90h]
  __int64 v51; // [rsp+40h] [rbp-90h]
  __int64 v52; // [rsp+60h] [rbp-70h] BYREF
  int v53; // [rsp+68h] [rbp-68h]
  __m128i v54; // [rsp+70h] [rbp-60h] BYREF
  __m128i v55; // [rsp+80h] [rbp-50h]
  unsigned __int8 *v56; // [rsp+90h] [rbp-40h]
  int v57; // [rsp+98h] [rbp-38h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(_QWORD *)(a2 + 48);
  v6 = *(_WORD *)(v5 + 16);
  v7 = *(_QWORD *)(v5 + 24);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v54, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v47 = v54.m128i_u16[4];
    v46 = v55.m128i_i64[0];
  }
  else
  {
    v47 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v46 = v37;
  }
  v9 = *(__int16 **)(a2 + 48);
  v10 = *a1;
  v11 = *v9;
  v49 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  v48 = v11;
  v13 = *(_QWORD *)(v12 + 64);
  v44 = v11;
  v14 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v15 = sub_2E79000(*(__int64 **)(v12 + 40));
  v16 = v14(v10, v15, v13, v44, v49);
  v18 = *(_DWORD *)(a2 + 64);
  v19 = v16;
  v20 = *(_QWORD *)(a2 + 40);
  v21 = v17;
  v22 = _mm_loadu_si128((const __m128i *)v20);
  v54 = v22;
  v23 = _mm_loadu_si128((const __m128i *)(v20 + 40));
  v56 = 0;
  v57 = 0;
  v55 = v23;
  if ( v18 == 3 )
  {
    v43 = v17;
    v41 = v19;
    v50 = v48;
    v35 = sub_375B580((__int64)a1, *(_QWORD *)(v20 + 80), v22, *(_QWORD *)(v20 + 88), v48, v49);
    v19 = v41;
    v21 = v43;
    v18 = 3;
    v56 = v35;
    v57 = v36;
  }
  v24 = *(_QWORD *)(a2 + 80);
  v52 = v24;
  if ( v24 )
  {
    v40 = v19;
    v42 = v21;
    v45 = v18;
    sub_B96E90((__int64)&v52, v24, 1);
    v19 = v40;
    v21 = v42;
    v18 = v45;
  }
  v25 = (__int64 *)a1[1];
  v26 = v18;
  v53 = *(_DWORD *)(a2 + 72);
  v27 = v50;
  LOWORD(v27) = v48;
  v51 = v27;
  v28 = (unsigned int *)sub_33E5110(v25, v27, v49, v19, v21);
  *((_QWORD *)&v39 + 1) = v26;
  *(_QWORD *)&v39 = &v54;
  v31 = sub_3411630(v25, *(unsigned int *)(a2 + 24), (__int64)&v52, v28, v29, v30, v39);
  sub_3760E70((__int64)a1, a2, 0, (unsigned __int64)v31, v32);
  *((_QWORD *)&v38 + 1) = v49;
  *(_QWORD *)&v38 = v51;
  v33 = sub_33FB620(a1[1], (__int64)v31, 1u, (__int64)&v52, v47, v46, v22, v38);
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
  return v33;
}
