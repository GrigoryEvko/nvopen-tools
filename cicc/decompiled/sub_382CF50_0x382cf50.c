// Function: sub_382CF50
// Address: 0x382cf50
//
unsigned __int8 *__fastcall sub_382CF50(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r12
  unsigned __int16 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  unsigned __int16 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdi
  int v16; // edx
  int v17; // r9d
  unsigned __int8 *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int16 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rdx
  const __m128i *v30; // rax
  _QWORD *v31; // r14
  __int128 v32; // rax
  __int64 v33; // r9
  char v35; // al
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  _QWORD *v40; // r14
  __int128 v41; // rax
  __int64 v42; // r9
  __int128 v43; // [rsp-30h] [rbp-F0h]
  __int128 v44; // [rsp-20h] [rbp-E0h]
  __int128 v45; // [rsp+0h] [rbp-C0h]
  __int128 v46; // [rsp+10h] [rbp-B0h]
  __int128 v47; // [rsp+20h] [rbp-A0h]
  unsigned __int16 v48; // [rsp+30h] [rbp-90h] BYREF
  __int64 v49; // [rsp+38h] [rbp-88h]
  unsigned int v50; // [rsp+40h] [rbp-80h] BYREF
  __int64 v51; // [rsp+48h] [rbp-78h]
  __int64 v52; // [rsp+50h] [rbp-70h] BYREF
  int v53; // [rsp+58h] [rbp-68h]
  unsigned __int16 v54; // [rsp+60h] [rbp-60h] BYREF
  __int64 v55; // [rsp+68h] [rbp-58h]
  __int64 v56; // [rsp+70h] [rbp-50h]
  __int64 v57; // [rsp+78h] [rbp-48h]
  __int64 v58; // [rsp+80h] [rbp-40h] BYREF
  __int64 v59; // [rsp+88h] [rbp-38h]

  v5 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 80);
  v7 = v5;
  v8 = *(unsigned __int16 **)(a2 + 48);
  v10 = v9;
  v11 = *v8;
  v49 = *((_QWORD *)v8 + 1);
  v12 = *(_QWORD *)(v7 + 48) + 16LL * (unsigned int)v10;
  v48 = v11;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v52 = v6;
  LOWORD(v50) = v13;
  v51 = v14;
  if ( v6 )
  {
    sub_B96E90((__int64)&v52, v6, 1);
    v13 = v50;
    v11 = v48;
  }
  v53 = *(_DWORD *)(a2 + 72);
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
      goto LABEL_11;
  }
  else if ( sub_30070B0((__int64)&v48) )
  {
    goto LABEL_11;
  }
  v15 = *a1;
  if ( v13 == 1 )
  {
    v35 = *(_BYTE *)(v15 + 7111);
    v16 = 1;
    if ( !v35 )
    {
LABEL_26:
      LOWORD(v58) = v13;
      v59 = v51;
      goto LABEL_27;
    }
LABEL_36:
    if ( v35 != 4 && v35 != 1 )
      goto LABEL_8;
LABEL_11:
    if ( v13 )
    {
LABEL_30:
      v16 = v13;
      goto LABEL_31;
    }
LABEL_12:
    if ( !sub_30070B0((__int64)&v50) )
    {
      LOWORD(v58) = 0;
      v59 = v51;
LABEL_14:
      v56 = sub_3007260((__int64)&v58);
      v22 = v56;
      v57 = v23;
      goto LABEL_15;
    }
    v13 = sub_3009970((__int64)&v50, v6, v19, v20, v21);
    goto LABEL_33;
  }
  if ( !v13 || (v16 = v13, !*(_QWORD *)(v15 + 8LL * v13 + 112)) )
  {
LABEL_8:
    v6 = a2;
    if ( sub_345C4B0(v15, a2, a1[1]) )
    {
      v18 = sub_33FAF80(a1[1], 215, (__int64)&v52, v50, v51, v17, a3);
      goto LABEL_22;
    }
    v13 = v50;
    if ( (_WORD)v50 )
      goto LABEL_30;
    goto LABEL_12;
  }
  v35 = *(_BYTE *)(v15 + 500LL * v13 + 6611);
  if ( v35 )
    goto LABEL_36;
LABEL_31:
  if ( (unsigned __int16)(v13 - 17) > 0xD3u )
    goto LABEL_26;
  v13 = word_4456580[v16 - 1];
  v36 = 0;
LABEL_33:
  v59 = v36;
  v16 = v13;
  LOWORD(v58) = v13;
  if ( !v13 )
    goto LABEL_14;
LABEL_27:
  if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
    goto LABEL_49;
  v22 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
LABEL_15:
  v24 = v48;
  if ( v48 )
  {
    if ( (unsigned __int16)(v48 - 17) > 0xD3u )
    {
LABEL_17:
      v25 = v49;
      goto LABEL_18;
    }
    v25 = 0;
    v24 = word_4456580[v48 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v48) )
      goto LABEL_17;
    v24 = sub_3009970((__int64)&v48, v22, v37, v38, v39);
  }
LABEL_18:
  v54 = v24;
  v55 = v25;
  if ( !v24 )
  {
    v26 = sub_3007260((__int64)&v54);
    v58 = v26;
    v59 = v27;
    goto LABEL_20;
  }
  if ( v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
LABEL_49:
    BUG();
  v26 = *(_QWORD *)&byte_444C4A0[16 * v24 - 16];
LABEL_20:
  *(_QWORD *)&v47 = sub_3400E40(a1[1], (unsigned int)(v22 - v26), v50, v51, (__int64)&v52, a3);
  *((_QWORD *)&v47 + 1) = v29;
  if ( *(_DWORD *)(a2 + 24) == 197 )
  {
    v40 = (_QWORD *)a1[1];
    *(_QWORD *)&v41 = sub_33FAF80((__int64)v40, 197, (__int64)&v52, v50, v51, v28, a3);
    v18 = sub_3406EB0(v40, 0xC0u, (__int64)&v52, v50, v51, v42, v41, v47);
  }
  else
  {
    v30 = *(const __m128i **)(a2 + 40);
    v31 = (_QWORD *)a1[1];
    *(_QWORD *)&v45 = v30[2].m128i_i64[1];
    v46 = (__int128)_mm_loadu_si128(v30 + 5);
    *((_QWORD *)&v44 + 1) = v30[3].m128i_i64[0];
    *(_QWORD *)&v44 = v45;
    *((_QWORD *)&v43 + 1) = v10;
    *(_QWORD *)&v43 = v7;
    *((_QWORD *)&v45 + 1) = *((_QWORD *)&v44 + 1);
    *(_QWORD *)&v32 = sub_340F900(v31, 0x19Du, (__int64)&v52, v50, v51, v28, v43, v44, v46);
    v18 = sub_33FC130(v31, 398, (__int64)&v52, v50, v51, v33, v32, v47, v45, v46);
  }
LABEL_22:
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
  return v18;
}
