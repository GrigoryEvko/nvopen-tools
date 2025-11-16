// Function: sub_382DCF0
// Address: 0x382dcf0
//
unsigned __int8 *__fastcall sub_382DCF0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r10
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  int v10; // r10d
  __int64 v11; // rax
  __int128 v12; // xmm0
  unsigned __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // r9
  unsigned int v16; // r10d
  unsigned __int8 *v17; // r14
  __int128 v19; // rax
  __int64 v20; // rcx
  bool v21; // al
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // kr00_4
  __int64 v29; // rdx
  unsigned __int8 *v30; // rax
  unsigned int v31; // r10d
  unsigned int v32; // edx
  __int16 v33; // ax
  __int64 v34; // rdx
  __int128 v35; // [rsp-20h] [rbp-F0h]
  __int128 v36; // [rsp+0h] [rbp-D0h]
  __int16 v37; // [rsp+12h] [rbp-BEh]
  __int64 v38; // [rsp+18h] [rbp-B8h]
  __int16 v39; // [rsp+1Ah] [rbp-B6h]
  __int16 v40; // [rsp+1Ah] [rbp-B6h]
  __int16 v41; // [rsp+1Ah] [rbp-B6h]
  __int64 v42; // [rsp+30h] [rbp-A0h]
  unsigned __int16 v43; // [rsp+3Eh] [rbp-92h]
  __int64 v44; // [rsp+40h] [rbp-90h] BYREF
  int v45; // [rsp+48h] [rbp-88h]
  unsigned int v46; // [rsp+50h] [rbp-80h] BYREF
  __int64 v47; // [rsp+58h] [rbp-78h]
  unsigned __int16 v48; // [rsp+60h] [rbp-70h] BYREF
  __int64 v49; // [rsp+68h] [rbp-68h]
  unsigned __int64 v50; // [rsp+70h] [rbp-60h]
  __int64 v51; // [rsp+78h] [rbp-58h]
  unsigned __int64 v52; // [rsp+80h] [rbp-50h] BYREF
  __int64 v53; // [rsp+88h] [rbp-48h]
  __int64 v54; // [rsp+90h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v44 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v44, v3, 1);
  v4 = *a1;
  v45 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v52, v4, *(_QWORD *)(v9 + 64), v7, v8);
    v43 = v53;
    v10 = (unsigned __int16)v53;
    v42 = v54;
  }
  else
  {
    v28 = v5(v4, *(_QWORD *)(v9 + 64), v7, v8);
    HIWORD(v10) = HIWORD(v28);
    v43 = v28;
    v42 = v29;
  }
  v11 = *(_QWORD *)(a2 + 40);
  v39 = HIWORD(v10);
  v12 = (__int128)_mm_loadu_si128((const __m128i *)(v11 + 40));
  v13 = *(_QWORD *)v11;
  v14 = *(_QWORD *)(v11 + 8);
  sub_2FE6CC0(
    (__int64)&v52,
    *a1,
    *(_QWORD *)(a1[1] + 64),
    *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v11 + 48LL) + 16LL * *(unsigned int *)(v11 + 8)),
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 48LL) + 16LL * *(unsigned int *)(v11 + 8) + 8));
  HIWORD(v16) = v39;
  if ( (_BYTE)v52 != 1 )
    goto LABEL_6;
  *(_QWORD *)&v19 = sub_37AE0F0((__int64)a1, v13, v14);
  HIWORD(v16) = v39;
  v36 = v19;
  *(_QWORD *)&v19 = *(_QWORD *)(v19 + 48) + 16LL * DWORD2(v19);
  v20 = *(_QWORD *)(v19 + 8);
  LOWORD(v52) = *(_WORD *)v19;
  v53 = v20;
  if ( (_WORD)v52 )
  {
    if ( (unsigned __int16)(v52 - 17) > 0xD3u )
    {
      LOWORD(v46) = v52;
      v47 = v20;
      if ( (_WORD)v52 == v43 )
        goto LABEL_20;
      goto LABEL_13;
    }
    v24 = 0;
    v33 = word_4456580[(unsigned __int16)v52 - 1];
  }
  else
  {
    v37 = v39;
    v38 = v20;
    v21 = sub_30070B0((__int64)&v52);
    v24 = v38;
    HIWORD(v16) = v37;
    if ( !v21 )
    {
      v47 = v38;
      LOWORD(v46) = 0;
      if ( v43 )
        goto LABEL_13;
LABEL_24:
      if ( v42 == v24 )
        goto LABEL_20;
      goto LABEL_13;
    }
    v33 = sub_3009970((__int64)&v52, v13, v22, v38, v23);
    HIWORD(v16) = v37;
    v24 = v34;
  }
  LOWORD(v46) = v33;
  v47 = v24;
  if ( v43 == v33 )
  {
    if ( v43 )
      goto LABEL_20;
    goto LABEL_24;
  }
LABEL_13:
  v40 = HIWORD(v16);
  v48 = v43;
  v49 = v42;
  v52 = sub_2D5B750(&v48);
  v53 = v25;
  v26 = sub_2D5B750((unsigned __int16 *)&v46);
  HIWORD(v16) = v40;
  v51 = v27;
  v50 = v26;
  if ( !(_BYTE)v27 && (_BYTE)v53 || v50 < v52 )
  {
LABEL_6:
    LOWORD(v16) = v43;
    *((_QWORD *)&v35 + 1) = v14;
    *(_QWORD *)&v35 = v13;
    v17 = sub_3406EB0((_QWORD *)a1[1], 0x9Eu, (__int64)&v44, v16, v42, v15, v35, v12);
    goto LABEL_7;
  }
LABEL_20:
  v41 = HIWORD(v16);
  v30 = sub_3406EB0((_QWORD *)a1[1], 0x9Eu, (__int64)&v44, v46, v47, v15, v36, v12);
  HIWORD(v31) = v41;
  LOWORD(v31) = v43;
  v17 = sub_33FAFB0(a1[1], (__int64)v30, v32, (__int64)&v44, v31, v42, (__m128i)v12);
LABEL_7:
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v17;
}
