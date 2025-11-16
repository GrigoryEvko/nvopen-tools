// Function: sub_37FCC90
// Address: 0x37fcc90
//
__int64 __fastcall sub_37FCC90(__int64 *a1, unsigned __int64 a2)
{
  int v3; // eax
  bool v4; // bl
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int); // r9
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r10
  __int64 v19; // rdx
  int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  _WORD *v24; // r11
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64, __int64, unsigned int); // r9
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // rdx
  __int64 (__fastcall *v32)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-D8h]
  __int64 v33; // [rsp+10h] [rbp-D0h]
  __int64 (__fastcall *v34)(__int64, __int64, unsigned int); // [rsp+18h] [rbp-C8h]
  int v35; // [rsp+18h] [rbp-C8h]
  __int64 v36; // [rsp+20h] [rbp-C0h]
  _WORD *v37; // [rsp+20h] [rbp-C0h]
  __int64 v38; // [rsp+28h] [rbp-B8h]
  __m128i v39; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v40; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+48h] [rbp-98h]
  __int64 v42; // [rsp+50h] [rbp-90h] BYREF
  int v43; // [rsp+58h] [rbp-88h]
  _QWORD v44[4]; // [rsp+60h] [rbp-80h] BYREF
  __int16 *v45; // [rsp+80h] [rbp-60h] BYREF
  __int64 v46; // [rsp+88h] [rbp-58h]
  __int64 (__fastcall *v47)(__int64, __int64, unsigned int); // [rsp+90h] [rbp-50h]
  __int64 v48; // [rsp+98h] [rbp-48h]
  __int64 v49; // [rsp+A0h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 > 239 )
  {
    v4 = (unsigned int)(v3 - 242) <= 1;
  }
  else
  {
    v4 = 1;
    if ( v3 <= 237 )
      v4 = (unsigned int)(v3 - 101) <= 0x2F;
  }
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v45, *a1, *(_QWORD *)(v9 + 64), v7, v8);
    v10 = v47;
    v11 = (unsigned __int16)v46;
  }
  else
  {
    v11 = v5(*a1, *(_QWORD *)(v9 + 64), v7, v8);
    v10 = v31;
  }
  v12 = *(_QWORD *)(a2 + 40);
  if ( v4 )
  {
    v13 = 40;
    v39 = _mm_loadu_si128((const __m128i *)(v12 + 40));
    v14 = *(_QWORD *)v12;
    v15 = *(_QWORD *)(v12 + 8);
  }
  else
  {
    v14 = 0;
    v13 = 0;
    v39 = _mm_loadu_si128((const __m128i *)v12);
    v15 = 0;
  }
  v34 = v10;
  v36 = v11;
  v38 = v13;
  v16 = *(_QWORD *)(v39.m128i_i64[0] + 48) + 16LL * v39.m128i_u32[2];
  v17 = sub_2FE5850(*(_WORD *)v16, *(_QWORD *)(v16 + 8), **(_WORD **)(a2 + 48));
  v18 = *(_QWORD *)(a2 + 40) + v38;
  v19 = *(_QWORD *)v18;
  v20 = v17;
  v21 = *(unsigned int *)(v18 + 8);
  v45 = &v40;
  v22 = *(_QWORD *)(a2 + 80);
  LOBYTE(v49) = 20;
  v23 = *(_QWORD *)(v19 + 48) + 16 * v21;
  v46 = 1;
  LOWORD(v19) = *(_WORD *)v23;
  v24 = (_WORD *)*a1;
  v25 = v36;
  v41 = *(_QWORD *)(v23 + 8);
  v26 = *(_QWORD *)(a2 + 48);
  v40 = v19;
  v27 = v34;
  LOWORD(v19) = *(_WORD *)v26;
  v28 = *(_QWORD *)(v26 + 8);
  v42 = v22;
  LOWORD(v47) = v19;
  v48 = v28;
  if ( v22 )
  {
    v32 = v34;
    v33 = v36;
    v35 = v20;
    v37 = v24;
    sub_B96E90((__int64)&v42, v22, 1);
    v27 = v32;
    v25 = v33;
    v20 = v35;
    v24 = v37;
  }
  v29 = a1[1];
  v43 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v44,
    v24,
    v29,
    v20,
    v25,
    v27,
    (__int64)&v39,
    1u,
    (__int64)v45,
    v46,
    (unsigned int)v47,
    v48,
    v49,
    (__int64)&v42,
    v14,
    v15);
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  if ( v4 )
    sub_3760E70((__int64)a1, a2, 1, v44[2], v44[3]);
  return v44[0];
}
