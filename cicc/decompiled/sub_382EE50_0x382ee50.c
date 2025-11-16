// Function: sub_382EE50
// Address: 0x382ee50
//
unsigned __int8 *__fastcall sub_382EE50(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int16 *v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int16 v13; // r14
  __int64 v14; // r12
  unsigned int v15; // edx
  unsigned __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rdx
  unsigned __int16 *v20; // rdx
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdx
  unsigned __int8 *v27; // r10
  __int64 v28; // rdx
  __int64 v29; // r9
  __int128 v30; // rax
  __int64 v31; // r9
  unsigned __int8 *v32; // r12
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int128 v38; // [rsp-40h] [rbp-100h]
  __int128 v39; // [rsp-30h] [rbp-F0h]
  __int128 v40; // [rsp-30h] [rbp-F0h]
  __int64 v41; // [rsp+0h] [rbp-C0h]
  __int128 v42; // [rsp+10h] [rbp-B0h]
  __int64 v43; // [rsp+30h] [rbp-90h] BYREF
  int v44; // [rsp+38h] [rbp-88h]
  unsigned int v45; // [rsp+40h] [rbp-80h] BYREF
  __int64 v46; // [rsp+48h] [rbp-78h]
  __int16 v47; // [rsp+50h] [rbp-70h] BYREF
  __int64 v48; // [rsp+58h] [rbp-68h]
  __int16 v49; // [rsp+60h] [rbp-60h] BYREF
  __int64 v50; // [rsp+68h] [rbp-58h]
  __int64 v51; // [rsp+70h] [rbp-50h]
  __int64 v52; // [rsp+78h] [rbp-48h]
  __int64 v53; // [rsp+80h] [rbp-40h] BYREF
  __int64 v54; // [rsp+88h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v43 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v43, v4, 1);
  v44 = *(_DWORD *)(a2 + 72);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v46 = *((_QWORD *)v5 + 1);
  v7 = *(_QWORD *)(a2 + 40);
  LOWORD(v45) = v6;
  v8 = sub_37AE0F0(a1, *(_QWORD *)v7, *(_QWORD *)(v7 + 8));
  v10 = v9;
  *((_QWORD *)&v39 + 1) = v9;
  *(_QWORD *)&v39 = v8;
  v12 = sub_340F900(
          *(_QWORD **)(a1 + 8),
          0x1CBu,
          (__int64)&v43,
          v45,
          v46,
          v11,
          v39,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  v13 = v45;
  v14 = v12;
  v16 = v15 | v10 & 0xFFFFFFFF00000000LL;
  if ( (_WORD)v45 )
  {
    if ( (unsigned __int16)(v45 - 17) <= 0xD3u )
    {
      v54 = 0;
      v13 = word_4456580[(unsigned __int16)v45 - 1];
      LOWORD(v53) = v13;
      if ( !v13 )
        goto LABEL_7;
      goto LABEL_21;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v45) )
  {
LABEL_5:
    v17 = v46;
    goto LABEL_6;
  }
  v13 = sub_3009970((__int64)&v45, 459, v35, v36, v37);
LABEL_6:
  LOWORD(v53) = v13;
  v54 = v17;
  if ( !v13 )
  {
LABEL_7:
    v51 = sub_3007260((__int64)&v53);
    LODWORD(v18) = v51;
    v52 = v19;
    goto LABEL_8;
  }
LABEL_21:
  if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
    goto LABEL_30;
  v18 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
LABEL_8:
  v20 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  v49 = v21;
  v50 = v22;
  if ( (_WORD)v21 )
  {
    if ( (unsigned __int16)(v21 - 17) > 0xD3u )
    {
      v47 = v21;
      v48 = v22;
LABEL_11:
      if ( (_WORD)v21 != 1 && (unsigned __int16)(v21 - 504) > 7u )
      {
        v23 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v21 - 16];
        goto LABEL_17;
      }
LABEL_30:
      BUG();
    }
    LOWORD(v21) = word_4456580[v21 - 1];
    v34 = 0;
  }
  else
  {
    v41 = v22;
    if ( !sub_30070B0((__int64)&v49) )
    {
      v48 = v41;
      v47 = 0;
      goto LABEL_16;
    }
    LOWORD(v21) = sub_3009970((__int64)&v49, 459, v41, v24, v25);
  }
  v47 = v21;
  v48 = v34;
  if ( (_WORD)v21 )
    goto LABEL_11;
LABEL_16:
  v23 = sub_3007260((__int64)&v47);
  v53 = v23;
  v54 = v26;
LABEL_17:
  v27 = sub_3400E40(*(_QWORD *)(a1 + 8), (unsigned int)(v18 - v23), v45, v46, (__int64)&v43, a3);
  *((_QWORD *)&v42 + 1) = v28;
  *(_QWORD *)&v42 = v27;
  *((_QWORD *)&v40 + 1) = v28;
  *(_QWORD *)&v40 = v27;
  *((_QWORD *)&v38 + 1) = v16;
  *(_QWORD *)&v38 = v14;
  *(_QWORD *)&v30 = sub_33FC130(
                      *(_QWORD **)(a1 + 8),
                      402,
                      (__int64)&v43,
                      v45,
                      v46,
                      v29,
                      v38,
                      v40,
                      *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  v32 = sub_33FC130(
          *(_QWORD **)(a1 + 8),
          397,
          (__int64)&v43,
          v45,
          v46,
          v31,
          v30,
          v42,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  return v32;
}
