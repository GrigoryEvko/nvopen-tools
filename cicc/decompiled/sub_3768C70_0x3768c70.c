// Function: sub_3768C70
// Address: 0x3768c70
//
unsigned __int8 *__fastcall sub_3768C70(__int64 *a1, __int64 a2)
{
  __int16 *v4; // rax
  __int16 v5; // dx
  __int64 v6; // rax
  unsigned int v7; // eax
  int v8; // r9d
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *result; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int128 v17; // xmm0
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // rdi
  unsigned __int8 *v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdx
  unsigned __int16 v24; // dx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r8
  __int128 v29; // rax
  __int64 v30; // r9
  int v31; // r9d
  bool v32; // al
  __int64 v33; // rcx
  __int64 v34; // r8
  unsigned __int16 v35; // ax
  __int64 v36; // rdx
  __int64 v37; // r8
  __int128 v38; // [rsp-10h] [rbp-D0h]
  __int64 v39; // [rsp-8h] [rbp-C8h]
  __int64 v40; // [rsp+0h] [rbp-C0h]
  unsigned int v41; // [rsp+Ch] [rbp-B4h]
  __int128 v42; // [rsp+10h] [rbp-B0h]
  __int128 v43; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v44; // [rsp+30h] [rbp-90h]
  unsigned int v45; // [rsp+40h] [rbp-80h] BYREF
  __int64 v46; // [rsp+48h] [rbp-78h]
  unsigned int v47; // [rsp+50h] [rbp-70h] BYREF
  __int64 v48; // [rsp+58h] [rbp-68h]
  __int64 v49; // [rsp+60h] [rbp-60h] BYREF
  int v50; // [rsp+68h] [rbp-58h]
  unsigned __int64 v51; // [rsp+70h] [rbp-50h] BYREF
  __int64 v52; // [rsp+78h] [rbp-48h]
  __int64 v53; // [rsp+80h] [rbp-40h]
  __int64 v54; // [rsp+88h] [rbp-38h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v45) = v5;
  v46 = v6;
  v7 = sub_327FDF0((unsigned __int16 *)&v45, a2);
  v9 = a1[1];
  v11 = v10;
  v12 = (unsigned __int16)v7;
  v47 = v7;
  v13 = 1;
  v48 = v11;
  if ( (_WORD)v12 != 1 )
  {
    if ( !(_WORD)v12 )
      return 0;
    v13 = (unsigned __int16)v12;
    if ( !*(_QWORD *)(v9 + 8 * v12 + 112) )
      return 0;
  }
  if ( (*(_BYTE *)(v9 + 500 * v13 + 6821) & 0xFB) != 0 )
    return 0;
  v15 = *(_QWORD *)(a2 + 40);
  v16 = *(_QWORD *)(a2 + 80);
  v17 = (__int128)_mm_loadu_si128((const __m128i *)(v15 + 40));
  v18 = *(_QWORD *)(v15 + 80);
  v49 = v16;
  v19 = *(_QWORD *)(v15 + 88);
  if ( v16 )
  {
    sub_B96E90((__int64)&v49, v16, 1);
    v15 = *(_QWORD *)(a2 + 40);
  }
  v20 = *a1;
  v50 = *(_DWORD *)(a2 + 72);
  v39 = *(_QWORD *)(v15 + 8);
  v21 = sub_33FAF80(v20, 234, (__int64)&v49, v47, v48, v8, (__m128i)v17);
  v22 = *a1;
  *((_QWORD *)&v43 + 1) = v23;
  v24 = v47;
  *(_QWORD *)&v43 = v21;
  if ( (_WORD)v47 )
  {
    if ( (unsigned __int16)(v47 - 17) <= 0xD3u )
    {
      v24 = word_4456580[(unsigned __int16)v47 - 1];
      v52 = 0;
      LOWORD(v51) = v24;
      if ( !v24 )
        goto LABEL_13;
      goto LABEL_27;
    }
    goto LABEL_11;
  }
  v32 = sub_30070B0((__int64)&v47);
  v24 = 0;
  if ( !v32 )
  {
LABEL_11:
    v25 = v48;
    goto LABEL_12;
  }
  v35 = sub_3009970((__int64)&v47, 234, 0, v33, v34);
  v37 = v36;
  v24 = v35;
  v25 = v37;
LABEL_12:
  LOWORD(v51) = v24;
  v52 = v25;
  if ( !v24 )
  {
LABEL_13:
    v26 = sub_3007260((__int64)&v51);
    v53 = v26;
    v54 = v27;
    goto LABEL_14;
  }
LABEL_27:
  if ( v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
    BUG();
  v26 = *(_QWORD *)&byte_444C4A0[16 * v24 - 16];
LABEL_14:
  LODWORD(v52) = v26;
  v28 = 1LL << ((unsigned __int8)v26 - 1);
  if ( (unsigned int)v26 <= 0x40 )
  {
    v51 = 0;
LABEL_16:
    v51 |= v28;
    goto LABEL_17;
  }
  v40 = 1LL << ((unsigned __int8)v26 - 1);
  v41 = v26 - 1;
  sub_C43690((__int64)&v51, 0, 0);
  v28 = v40;
  if ( (unsigned int)v52 <= 0x40 )
    goto LABEL_16;
  *(_QWORD *)(v51 + 8LL * (v41 >> 6)) |= v40;
LABEL_17:
  *(_QWORD *)&v29 = sub_34007B0(v22, (__int64)&v51, (__int64)&v49, v47, v48, 0, (__m128i)v17, 0);
  v30 = v39;
  if ( (unsigned int)v52 > 0x40 && v51 )
  {
    v42 = v29;
    j_j___libc_free_0_0(v51);
    v29 = v42;
  }
  *((_QWORD *)&v38 + 1) = v19;
  *(_QWORD *)&v38 = v18;
  sub_33FC130((_QWORD *)*a1, 407, (__int64)&v49, v47, v48, v30, v43, v29, v17, v38);
  result = sub_33FAF80(*a1, 234, (__int64)&v49, v45, v46, v31, (__m128i)v17);
  if ( v49 )
  {
    v44 = result;
    sub_B91220((__int64)&v49, v49);
    return v44;
  }
  return result;
}
