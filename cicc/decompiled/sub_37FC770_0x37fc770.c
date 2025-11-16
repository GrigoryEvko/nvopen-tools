// Function: sub_37FC770
// Address: 0x37fc770
//
__int64 __fastcall sub_37FC770(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // rax
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int); // rcx
  unsigned int v5; // r11d
  const __m128i *v6; // rax
  _WORD *v7; // r15
  __m128i v8; // xmm0
  __int64 v9; // rsi
  __int64 v10; // rax
  __int16 v11; // dx
  __int16 *v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // r10
  __int16 *v16; // rdx
  unsigned __int16 v17; // ax
  __int64 v18; // r8
  __int64 (__fastcall *v19)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v20; // rdx
  unsigned int v21; // r15d
  __int64 (__fastcall *v22)(__int64, __int64, unsigned int); // r9
  int v23; // eax
  __int64 v24; // rsi
  int v25; // ecx
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int); // r9
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // rdx
  __int64 (__fastcall *v32)(__int64, __int64, unsigned int); // rdx
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int); // [rsp+0h] [rbp-F0h]
  __int64 (__fastcall *v34)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-E0h]
  unsigned int v35; // [rsp+18h] [rbp-D8h]
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-D8h]
  int v37; // [rsp+18h] [rbp-D8h]
  _WORD *v38; // [rsp+20h] [rbp-D0h]
  __m128i v39; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v40; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+48h] [rbp-A8h]
  __m128i v42; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v43; // [rsp+60h] [rbp-90h] BYREF
  int v44; // [rsp+68h] [rbp-88h]
  __m128i v45; // [rsp+70h] [rbp-80h] BYREF
  __int64 (__fastcall *v46)(__int64, __int64, unsigned int); // [rsp+80h] [rbp-70h]
  __int16 *v47; // [rsp+90h] [rbp-60h] BYREF
  __int64 v48; // [rsp+98h] [rbp-58h]
  __int64 (__fastcall *v49)(__int64, __int64, unsigned int); // [rsp+A0h] [rbp-50h]
  __int64 v50; // [rsp+A8h] [rbp-48h]
  __int64 v51; // [rsp+B0h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v47, *a1, *(_QWORD *)(a1[1] + 64), 12, 0);
    v4 = v49;
    v5 = (unsigned __int16)v48;
  }
  else
  {
    v5 = v3(*a1, *(_QWORD *)(a1[1] + 64), 12u, 0);
    v4 = v31;
  }
  v6 = *(const __m128i **)(a2 + 40);
  LOBYTE(v51) = 20;
  v48 = 1;
  v7 = (_WORD *)*a1;
  v8 = _mm_loadu_si128(v6);
  v47 = &v40;
  v9 = *(_QWORD *)(a2 + 80);
  v39 = v8;
  v10 = *(_QWORD *)(v6->m128i_i64[0] + 48) + 16LL * v6->m128i_u32[2];
  v11 = *(_WORD *)v10;
  v41 = *(_QWORD *)(v10 + 8);
  v12 = *(__int16 **)(a2 + 48);
  v40 = v11;
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v43 = v9;
  LOWORD(v49) = v13;
  v50 = v14;
  if ( v9 )
  {
    v33 = v4;
    v35 = v5;
    sub_B96E90((__int64)&v43, v9, 1);
    v4 = v33;
    v5 = v35;
  }
  v15 = a1[1];
  v44 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)&v45,
    v7,
    v15,
    346,
    v5,
    v4,
    (__int64)&v39,
    1u,
    (__int64)v47,
    v48,
    (unsigned int)v49,
    v50,
    v51,
    (__int64)&v43,
    0,
    0);
  v42 = _mm_loadu_si128(&v45);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  v16 = *(__int16 **)(a2 + 48);
  v17 = *v16;
  if ( *v16 == 12 )
    return v42.m128i_i64[0];
  v18 = *((_QWORD *)v16 + 1);
  v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v20 = a1[1];
  if ( v19 == sub_2D56A50 )
  {
    HIWORD(v21) = 0;
    sub_2FE6CC0((__int64)&v45, *a1, *(_QWORD *)(v20 + 64), v17, v18);
    LOWORD(v21) = v45.m128i_i16[4];
    v22 = v46;
  }
  else
  {
    v21 = v19(*a1, *(_QWORD *)(v20 + 64), v17, v18);
    v22 = v32;
  }
  v36 = v22;
  v23 = sub_2FE5770(12, 0, **(_WORD **)(a2 + 48));
  v24 = *(_QWORD *)(a2 + 80);
  v25 = v23;
  v26 = *a1;
  v27 = v36;
  v43 = v24;
  v38 = (_WORD *)v26;
  if ( v24 )
  {
    v34 = v36;
    v37 = v25;
    sub_B96E90((__int64)&v43, v24, 1);
    v27 = v34;
    v25 = v37;
  }
  v28 = a1[1];
  v44 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)&v45,
    v38,
    v28,
    v25,
    v21,
    v27,
    (__int64)&v42,
    1u,
    (__int64)v47,
    v48,
    (unsigned int)v49,
    v50,
    v51,
    (__int64)&v43,
    0,
    0);
  v29 = v45.m128i_i64[0];
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  return v29;
}
