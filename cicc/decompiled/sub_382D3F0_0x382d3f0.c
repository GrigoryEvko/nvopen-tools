// Function: sub_382D3F0
// Address: 0x382d3f0
//
unsigned __int8 *__fastcall sub_382D3F0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r12
  unsigned __int16 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rdx
  unsigned __int16 v12; // cx
  __int64 v13; // rax
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  __int64 v16; // rdi
  char v17; // cl
  int v18; // edx
  int v19; // r9d
  unsigned __int8 *v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rdx
  unsigned __int16 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rdx
  const __m128i *v32; // rax
  _QWORD *v33; // r14
  __int128 v34; // rax
  __int64 v35; // r9
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // rdx
  _QWORD *v42; // r14
  __int128 v43; // rax
  __int64 v44; // r9
  __int128 v45; // [rsp-30h] [rbp-F0h]
  __int128 v46; // [rsp-20h] [rbp-E0h]
  __int128 v47; // [rsp+0h] [rbp-C0h]
  __int128 v48; // [rsp+10h] [rbp-B0h]
  __int128 v49; // [rsp+20h] [rbp-A0h]
  unsigned __int16 v50; // [rsp+30h] [rbp-90h] BYREF
  __int64 v51; // [rsp+38h] [rbp-88h]
  unsigned int v52; // [rsp+40h] [rbp-80h] BYREF
  __int64 v53; // [rsp+48h] [rbp-78h]
  __int64 v54; // [rsp+50h] [rbp-70h] BYREF
  int v55; // [rsp+58h] [rbp-68h]
  unsigned __int16 v56; // [rsp+60h] [rbp-60h] BYREF
  __int64 v57; // [rsp+68h] [rbp-58h]
  __int64 v58; // [rsp+70h] [rbp-50h]
  __int64 v59; // [rsp+78h] [rbp-48h]
  __int64 v60; // [rsp+80h] [rbp-40h] BYREF
  __int64 v61; // [rsp+88h] [rbp-38h]

  v5 = sub_37AE0F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 80);
  v7 = v5;
  v8 = *(unsigned __int16 **)(a2 + 48);
  v10 = v9;
  v11 = *(_QWORD *)(v7 + 48) + 16LL * (unsigned int)v9;
  v12 = *v8;
  v13 = *((_QWORD *)v8 + 1);
  v50 = v12;
  v51 = v13;
  v14 = *(_WORD *)v11;
  v15 = *(_QWORD *)(v11 + 8);
  v54 = v6;
  LOWORD(v52) = v14;
  v53 = v15;
  if ( v6 )
  {
    sub_B96E90((__int64)&v54, v6, 1);
    v14 = v52;
    v12 = v50;
  }
  v55 = *(_DWORD *)(a2 + 72);
  if ( v12 && (unsigned __int16)(v12 - 17) > 0xD3u )
  {
    v16 = *a1;
    if ( v14 == 1 )
    {
      v17 = *(_BYTE *)(v16 + 7115);
      v18 = 1;
      if ( !v17 )
      {
LABEL_39:
        LOWORD(v60) = v14;
        v61 = v53;
        goto LABEL_30;
      }
    }
    else
    {
      if ( !v14 )
        goto LABEL_9;
      v18 = v14;
      if ( !*(_QWORD *)(v16 + 8LL * v14 + 112) )
        goto LABEL_9;
      v17 = *(_BYTE *)(v16 + 500LL * v14 + 6615);
      if ( !v17 )
        goto LABEL_27;
    }
    if ( v17 != 4 && v17 != 1 )
    {
LABEL_9:
      v6 = a2;
      if ( sub_345E690(v16, a2, (_QWORD *)a1[1]) )
      {
        v20 = sub_33FAF80(a1[1], 215, (__int64)&v54, v52, v53, v19, a3);
        goto LABEL_23;
      }
      v14 = v52;
    }
  }
  if ( !v14 )
  {
    if ( !sub_30070B0((__int64)&v52) )
    {
      LOWORD(v60) = 0;
      v61 = v53;
LABEL_15:
      v58 = sub_3007260((__int64)&v60);
      v24 = v58;
      v59 = v25;
      goto LABEL_16;
    }
    v14 = sub_3009970((__int64)&v52, v6, v21, v22, v23);
    goto LABEL_29;
  }
  v18 = v14;
LABEL_27:
  if ( (unsigned __int16)(v14 - 17) > 0xD3u )
    goto LABEL_39;
  v14 = word_4456580[v18 - 1];
  v37 = 0;
LABEL_29:
  v61 = v37;
  v18 = v14;
  LOWORD(v60) = v14;
  if ( !v14 )
    goto LABEL_15;
LABEL_30:
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
LABEL_46:
    BUG();
  v24 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
LABEL_16:
  v26 = v50;
  if ( !v50 )
  {
    if ( sub_30070B0((__int64)&v50) )
    {
      v26 = sub_3009970((__int64)&v50, v24, v38, v39, v40);
      v57 = v41;
      v56 = v26;
      if ( !v26 )
        goto LABEL_20;
      goto LABEL_35;
    }
    goto LABEL_18;
  }
  if ( (unsigned __int16)(v50 - 17) > 0xD3u )
  {
LABEL_18:
    v27 = v51;
    goto LABEL_19;
  }
  v27 = 0;
  v26 = word_4456580[v50 - 1];
LABEL_19:
  v56 = v26;
  v57 = v27;
  if ( !v26 )
  {
LABEL_20:
    v28 = sub_3007260((__int64)&v56);
    v60 = v28;
    v61 = v29;
    goto LABEL_21;
  }
LABEL_35:
  if ( v26 == 1 || (unsigned __int16)(v26 - 504) <= 7u )
    goto LABEL_46;
  v28 = *(_QWORD *)&byte_444C4A0[16 * v26 - 16];
LABEL_21:
  *(_QWORD *)&v49 = sub_3400E40(a1[1], (unsigned int)(v24 - v28), v52, v53, (__int64)&v54, a3);
  *((_QWORD *)&v49 + 1) = v31;
  if ( *(_DWORD *)(a2 + 24) == 201 )
  {
    v42 = (_QWORD *)a1[1];
    *(_QWORD *)&v43 = sub_33FAF80((__int64)v42, 201, (__int64)&v54, v52, v53, v30, a3);
    v20 = sub_3406EB0(v42, 0xC0u, (__int64)&v54, v52, v53, v44, v43, v49);
  }
  else
  {
    v32 = *(const __m128i **)(a2 + 40);
    v33 = (_QWORD *)a1[1];
    *(_QWORD *)&v47 = v32[2].m128i_i64[1];
    v48 = (__int128)_mm_loadu_si128(v32 + 5);
    *((_QWORD *)&v46 + 1) = v32[3].m128i_i64[0];
    *(_QWORD *)&v46 = v47;
    *((_QWORD *)&v45 + 1) = v10;
    *(_QWORD *)&v45 = v7;
    *((_QWORD *)&v47 + 1) = *((_QWORD *)&v46 + 1);
    *(_QWORD *)&v34 = sub_340F900(v33, 0x19Eu, (__int64)&v54, v52, v53, v30, v45, v46, v48);
    v20 = sub_33FC130(v33, 398, (__int64)&v54, v52, v53, v35, v34, v49, v47, v48);
  }
LABEL_23:
  if ( v54 )
    sub_B91220((__int64)&v54, v54);
  return v20;
}
