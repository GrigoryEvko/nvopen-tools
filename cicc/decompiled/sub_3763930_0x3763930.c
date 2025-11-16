// Function: sub_3763930
// Address: 0x3763930
//
unsigned __int8 *__fastcall sub_3763930(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int128 v8; // xmm0
  unsigned __int16 *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r13
  unsigned int v18; // r12d
  _BYTE *v19; // rax
  unsigned __int8 *v20; // r13
  unsigned __int16 *v22; // rdx
  int v23; // eax
  __int64 v24; // rdx
  __int16 v25; // ax
  __int128 v26; // rax
  __int64 v27; // r9
  __int128 v28; // rax
  __int64 v29; // r9
  unsigned int v30; // edx
  unsigned __int8 *v31; // rax
  unsigned int v32; // edx
  __int128 v33; // [rsp-30h] [rbp-100h]
  __int128 v34; // [rsp-10h] [rbp-E0h]
  __int128 v35; // [rsp-10h] [rbp-E0h]
  __int128 v36; // [rsp-10h] [rbp-E0h]
  __int128 v37; // [rsp-10h] [rbp-E0h]
  __int128 v38; // [rsp+0h] [rbp-D0h]
  __int128 v39; // [rsp+10h] [rbp-C0h]
  __int128 v40; // [rsp+30h] [rbp-A0h]
  __int64 v41; // [rsp+30h] [rbp-A0h]
  __int64 v42; // [rsp+38h] [rbp-98h]
  __int128 v43; // [rsp+50h] [rbp-80h]
  __int64 v44; // [rsp+80h] [rbp-50h] BYREF
  int v45; // [rsp+88h] [rbp-48h]
  __int16 v46; // [rsp+90h] [rbp-40h] BYREF
  __int64 v47; // [rsp+98h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v44 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v44, v6, 1);
  v45 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = (__int128)_mm_loadu_si128((const __m128i *)v7);
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * *(unsigned int *)(v7 + 8));
  v10 = *(_QWORD *)(v7 + 40);
  v11 = *(unsigned int *)(v7 + 48);
  v12 = *(_QWORD *)(v7 + 80);
  v43 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v13 = *(_QWORD *)(v7 + 88);
  v14 = *(_QWORD *)(v7 + 120);
  v15 = *(_QWORD *)(v7 + 128);
  v16 = *v9;
  v17 = *((_QWORD *)v9 + 1);
  v18 = (unsigned __int16)v16;
  if ( !(_WORD)v16 )
    goto LABEL_7;
  v19 = (_BYTE *)(a1[1] + 500 * v16);
  if ( v19[6810] == 2 || v19[6821] == 2 || v19[6814] == 2 )
    goto LABEL_7;
  v22 = (unsigned __int16 *)(*(_QWORD *)(v10 + 48) + 16 * v11);
  v23 = *v22;
  v24 = *((_QWORD *)v22 + 1);
  v46 = v23;
  v47 = v24;
  if ( (_WORD)v23 )
  {
    v25 = word_4456580[v23 - 1];
  }
  else
  {
    v41 = v12;
    v42 = v13;
    v25 = sub_3009970((__int64)&v46, (__int64)v9, v24, v10, a5);
    v12 = v41;
    v13 = v42;
  }
  if ( v25 != 2 )
  {
LABEL_7:
    v20 = 0;
  }
  else
  {
    *(_QWORD *)&v38 = v12;
    *((_QWORD *)&v38 + 1) = v13;
    *(_QWORD *)&v26 = sub_34015B0(*a1, (__int64)&v44, v18, v17, 0, 0, (__m128i)v8);
    *((_QWORD *)&v34 + 1) = v15;
    *(_QWORD *)&v34 = v14;
    v40 = v26;
    *(_QWORD *)&v28 = sub_33FC130((_QWORD *)*a1, 407, (__int64)&v44, v18, v17, v27, v8, v26, v26, v34);
    *((_QWORD *)&v35 + 1) = v15;
    *(_QWORD *)&v35 = v14;
    v39 = v28;
    *(_QWORD *)&v43 = sub_33FC130((_QWORD *)*a1, 396, (__int64)&v44, v18, v17, v29, v43, v8, v40, v35);
    *((_QWORD *)&v36 + 1) = v15;
    *(_QWORD *)&v36 = v14;
    *((_QWORD *)&v43 + 1) = v30 | *((_QWORD *)&v43 + 1) & 0xFFFFFFFF00000000LL;
    v31 = sub_33FC130((_QWORD *)*a1, 396, (__int64)&v44, v18, v17, 0xFFFFFFFF00000000LL, v38, v39, v40, v36);
    *((_QWORD *)&v37 + 1) = v15;
    *(_QWORD *)&v37 = v14;
    *((_QWORD *)&v33 + 1) = v32 | *((_QWORD *)&v38 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v33 = v31;
    v20 = sub_33FC130((_QWORD *)*a1, 400, (__int64)&v44, v18, v17, 0xFFFFFFFF00000000LL, v43, v33, v40, v37);
  }
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v20;
}
