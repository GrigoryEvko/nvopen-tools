// Function: sub_37FD620
// Address: 0x37fd620
//
__int64 __fastcall sub_37FD620(__int64 a1, unsigned __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  unsigned int v7; // r13d
  int v9; // eax
  unsigned int v10; // r15d
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  unsigned __int16 v16; // di
  __int64 v17; // r14
  int v18; // ebx
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  char v21; // si
  _BYTE *v22; // rax
  char v23; // dl
  int v24; // eax
  unsigned __int8 *v25; // rax
  _WORD *v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r11
  __int64 (__fastcall *v30)(__int64, __int64, unsigned int, __int64); // rax
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // r9
  __int64 v32; // r8
  int v33; // r10d
  __int64 v34; // r12
  _BYTE *v36; // rdx
  __int64 v37; // rax
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int); // rdx
  __m128i v39; // [rsp+0h] [rbp-100h]
  int v40; // [rsp+14h] [rbp-ECh]
  bool v41; // [rsp+18h] [rbp-E8h]
  __int64 v42; // [rsp+20h] [rbp-E0h]
  bool v44; // [rsp+3Bh] [rbp-C5h]
  unsigned __int16 v45; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+48h] [rbp-B8h]
  __int64 v47; // [rsp+50h] [rbp-B0h] BYREF
  int v48; // [rsp+58h] [rbp-A8h]
  _QWORD v49[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+70h] [rbp-90h]
  __int64 v51; // [rsp+78h] [rbp-88h]
  __int64 v52; // [rsp+80h] [rbp-80h] BYREF
  __int64 v53; // [rsp+88h] [rbp-78h]
  __int64 (__fastcall *v54)(__int64, __int64, unsigned int); // [rsp+90h] [rbp-70h]
  __int64 v55; // [rsp+98h] [rbp-68h]
  unsigned __int16 *v56; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v57; // [rsp+A8h] [rbp-58h]
  __int64 v58; // [rsp+B0h] [rbp-50h]
  __int64 v59; // [rsp+B8h] [rbp-48h]
  __int64 v60; // [rsp+C0h] [rbp-40h]

  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 > 239 )
  {
    v42 = (unsigned int)(v9 - 242) < 2 ? 0x28 : 0;
    v44 = (unsigned int)(v9 - 242) < 2;
    goto LABEL_7;
  }
  if ( v9 > 237 || (unsigned int)(v9 - 101) <= 0x2F )
  {
    v42 = 40;
    v44 = 1;
    if ( v9 == 220 )
    {
      v41 = 1;
      goto LABEL_8;
    }
    goto LABEL_7;
  }
  v42 = 0;
  v44 = 0;
  if ( v9 != 220 )
  {
LABEL_7:
    v41 = v9 == 143;
    goto LABEL_8;
  }
  v41 = 1;
LABEL_8:
  HIWORD(v10) = 0;
  v11 = *(_QWORD *)(a2 + 80);
  v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v42) + 48LL)
      + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v42 + 8);
  v13 = *(_QWORD *)(a2 + 48);
  v14 = *(_WORD *)v12;
  v15 = *(_QWORD *)(v12 + 8);
  v45 = v14;
  v46 = v15;
  v16 = *(_WORD *)v13;
  v17 = *(_QWORD *)(v13 + 8);
  v47 = v11;
  if ( v11 )
  {
    sub_B96E90((__int64)&v47, v11, 1);
    v15 = v46;
    v14 = v45;
  }
  v18 = 2;
  v48 = *(_DWORD *)(a2 + 72);
  while ( 1 )
  {
    LOWORD(v10) = v18;
    if ( (_WORD)v18 != v14 )
    {
      LOWORD(v56) = v14;
      v57 = v15;
      if ( v14 )
      {
        if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
          BUG();
        v36 = &byte_444C4A0[16 * v14 - 16];
        v19 = *(_QWORD *)v36;
        v21 = v36[8];
      }
      else
      {
        v50 = sub_3007260((__int64)&v56);
        v19 = v50;
        v51 = v20;
        v21 = v20;
      }
      v22 = &byte_444C4A0[16 * v18 - 16];
      v23 = v22[8];
      if ( !v23 && v21 )
        goto LABEL_17;
      if ( v19 > *(_QWORD *)v22 )
      {
        v23 = 0;
LABEL_17:
        v24 = 729;
        goto LABEL_18;
      }
    }
    if ( v41 )
      break;
    v24 = sub_2FE5D50(v18, 0, v16);
    v23 = v24 != 729;
LABEL_18:
    if ( (unsigned int)++v18 > 9 )
      goto LABEL_24;
LABEL_19:
    if ( v23 )
      goto LABEL_24;
    v15 = v46;
    v14 = v45;
  }
  v24 = sub_2FE5C30(v18, 0, v16);
  v23 = v24 != 729;
  if ( (unsigned int)++v18 <= 9 )
    goto LABEL_19;
LABEL_24:
  if ( v44 )
  {
    a3 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v39 = a3;
  }
  else
  {
    v39 = 0u;
  }
  v40 = v24;
  v25 = sub_33FAF80(*(_QWORD *)(a1 + 8), (unsigned int)!v41 + 213, (__int64)&v47, v10, 0, a7, a3);
  v57 = 1;
  v49[0] = v25;
  v26 = *(_WORD **)a1;
  v56 = &v45;
  v49[1] = v27;
  v28 = *(_QWORD *)v26;
  v29 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL);
  LOWORD(v58) = v16;
  v30 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v28 + 592);
  v59 = v17;
  LOBYTE(v60) = v41 | 0x14;
  if ( v30 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v52, (__int64)v26, v29, v16, v17);
    v31 = v54;
    v32 = (unsigned __int16)v53;
    v33 = v40;
  }
  else
  {
    LOWORD(v7) = v16;
    v37 = v30((__int64)v26, v29, v7, v17);
    v33 = v40;
    v32 = v37;
    v31 = v38;
  }
  sub_3494590(
    (__int64)&v52,
    v26,
    *(_QWORD *)(a1 + 8),
    v33,
    v32,
    v31,
    (__int64)v49,
    1u,
    (__int64)v56,
    v57,
    v58,
    v59,
    v60,
    (__int64)&v47,
    v39.m128i_i64[0],
    v39.m128i_i64[1]);
  if ( v44 )
    sub_3760E70(a1, a2, 1, (unsigned __int64)v54, v55);
  v34 = v52;
  if ( v47 )
    sub_B91220((__int64)&v47, v47);
  return v34;
}
