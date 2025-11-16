// Function: sub_382AE60
// Address: 0x382ae60
//
unsigned __int8 *__fastcall sub_382AE60(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rsi
  int v12; // eax
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r9
  const __m128i *v16; // rbx
  unsigned int *v17; // r11
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  const __m128i *v23; // r12
  __m128i *v24; // rax
  int v25; // esi
  _QWORD *v26; // rcx
  _QWORD *v27; // rdi
  __int64 v28; // rdx
  unsigned __int8 *v29; // rax
  __int64 v30; // rdi
  unsigned __int8 *v31; // r12
  __int64 v33; // rdx
  __int128 v34; // [rsp-10h] [rbp-100h]
  __int64 v35; // [rsp+8h] [rbp-E8h]
  unsigned int *v36; // [rsp+10h] [rbp-E0h]
  int v37; // [rsp+18h] [rbp-D8h]
  __int64 v38; // [rsp+30h] [rbp-C0h] BYREF
  int v39; // [rsp+38h] [rbp-B8h]
  __int64 v40; // [rsp+40h] [rbp-B0h] BYREF
  int v41; // [rsp+48h] [rbp-A8h]
  __int64 v42; // [rsp+50h] [rbp-A0h]
  int v43; // [rsp+58h] [rbp-98h]
  unsigned __int8 *v44; // [rsp+60h] [rbp-90h] BYREF
  int v45; // [rsp+68h] [rbp-88h]
  unsigned __int8 *v46; // [rsp+70h] [rbp-80h]
  int v47; // [rsp+78h] [rbp-78h]
  _QWORD *v48; // [rsp+80h] [rbp-70h] BYREF
  __int64 v49; // [rsp+88h] [rbp-68h]
  _QWORD v50[2]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v51; // [rsp+A0h] [rbp-50h]
  __int64 v52; // [rsp+A8h] [rbp-48h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v48, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v9 = (unsigned __int16)v49;
    v10 = v50[0];
  }
  else
  {
    v9 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v10 = v33;
  }
  v11 = *(_QWORD *)(a2 + 80);
  v38 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v38, v11, 1);
  v12 = *(_DWORD *)(a2 + 72);
  v13 = (_QWORD *)a1[1];
  v51 = 262;
  v48 = (_QWORD *)v9;
  v49 = v10;
  LOWORD(v50[0]) = 1;
  v39 = v12;
  v50[1] = 0;
  v52 = 0;
  v14 = sub_33E5830(v13, (unsigned __int16 *)&v48, 3);
  v16 = *(const __m128i **)(a2 + 40);
  v17 = (unsigned int *)v14;
  v18 = *(unsigned int *)(a2 + 64);
  v20 = v19;
  v48 = v50;
  v21 = 5 * v18;
  v18 *= 40LL;
  v22 = 0xCCCCCCCCCCCCCCCDLL * v21;
  v23 = (const __m128i *)((char *)v16 + v18);
  v49 = 0x300000000LL;
  if ( v18 > 0x78 )
  {
    v35 = v20;
    v36 = v17;
    v37 = v22;
    sub_C8D5F0((__int64)&v48, v50, v22, 0x10u, v20, v15);
    v25 = v49;
    v26 = v48;
    LODWORD(v22) = v37;
    v17 = v36;
    v20 = v35;
    v24 = (__m128i *)&v48[2 * (unsigned int)v49];
  }
  else
  {
    v24 = (__m128i *)v50;
    v25 = 0;
    v26 = v50;
  }
  if ( v16 != v23 )
  {
    do
    {
      if ( v24 )
        *v24 = _mm_loadu_si128(v16);
      v16 = (const __m128i *)((char *)v16 + 40);
      ++v24;
    }
    while ( v23 != v16 );
    v26 = v48;
    v25 = v49;
  }
  v27 = (_QWORD *)a1[1];
  v28 = (unsigned int)(v22 + v25);
  LODWORD(v49) = v28;
  *((_QWORD *)&v34 + 1) = v28;
  *(_QWORD *)&v34 = v26;
  v29 = sub_3411630(v27, 394, (__int64)&v38, v17, v20, v15, v34);
  v30 = a1[1];
  v44 = v29;
  v31 = v29;
  v46 = v29;
  v40 = a2;
  v41 = 1;
  v42 = a2;
  v43 = 2;
  v45 = 1;
  v47 = 2;
  sub_3417D60(v30, &v40, (__int64 *)&v44, 2);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  return v31;
}
