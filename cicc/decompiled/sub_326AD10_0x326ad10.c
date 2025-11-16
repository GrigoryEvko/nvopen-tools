// Function: sub_326AD10
// Address: 0x326ad10
//
__int64 __fastcall sub_326AD10(const __m128i *a1, unsigned __int16 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int32 v8; // r15d
  __int64 v9; // rbx
  __int64 v10; // r14
  unsigned __int32 v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rcx
  int v15; // edi
  __int64 v16; // rax
  __int64 v17; // r9
  bool v19; // al
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int128 v26; // rax
  int v27; // r9d
  unsigned int v28; // ecx
  int v29; // edx
  int v30; // esi
  int v31; // r8d
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int16 v35; // ax
  __int128 v36; // rax
  int v37; // r9d
  unsigned int v38; // edx
  __int128 v39; // rax
  __int128 v40; // rax
  int v41; // r9d
  unsigned int v42; // edx
  __int128 v43; // rax
  __int128 v44; // [rsp-20h] [rbp-E0h]
  __int128 v45; // [rsp-10h] [rbp-D0h]
  __int64 v46; // [rsp+0h] [rbp-C0h]
  __int64 v47; // [rsp+10h] [rbp-B0h]
  __int32 v48; // [rsp+1Ch] [rbp-A4h]
  __int64 v49; // [rsp+20h] [rbp-A0h]
  __int64 v50; // [rsp+28h] [rbp-98h]
  __int128 v51; // [rsp+30h] [rbp-90h]
  __int64 v52; // [rsp+40h] [rbp-80h]
  __int128 v53; // [rsp+40h] [rbp-80h]
  __int64 v54; // [rsp+50h] [rbp-70h]
  __int128 v55; // [rsp+50h] [rbp-70h]
  unsigned int v56; // [rsp+60h] [rbp-60h] BYREF
  __int64 v57; // [rsp+68h] [rbp-58h]
  __int16 v58; // [rsp+70h] [rbp-50h] BYREF
  __int64 v59; // [rsp+78h] [rbp-48h]

  v8 = a1->m128i_u32[2];
  v9 = a1->m128i_i64[0];
  v10 = a1[2].m128i_i64[1];
  v54 = a1[3].m128i_i64[0];
  v11 = a1[3].m128i_u32[0];
  v48 = a1[5].m128i_i32[2];
  v12 = *((_QWORD *)a2 + 1);
  v51 = (__int128)_mm_loadu_si128(a1);
  v52 = a1[5].m128i_i64[0];
  v13 = a1[5].m128i_i64[1];
  v57 = v12;
  v50 = v13;
  v14 = a1[5].m128i_i64[0];
  v15 = *a2;
  v16 = *(_QWORD *)(v9 + 48) + 16LL * v8;
  v49 = v14;
  LOWORD(v56) = v15;
  if ( (_WORD)v15 != *(_WORD *)v16 )
    return 0;
  v17 = v10;
  if ( *(_QWORD *)(v16 + 8) != v12 )
  {
    if ( !(_WORD)v15 )
      return 0;
    goto LABEL_4;
  }
  if ( (_WORD)v15 )
  {
LABEL_4:
    if ( (unsigned __int16)(v15 - 17) > 0xD3u )
    {
      v58 = v15;
      v59 = v12;
LABEL_6:
      if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
        BUG();
      if ( *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16] != 1 )
        return 0;
      goto LABEL_14;
    }
    v32 = 0;
    LOWORD(v15) = word_4456580[v15 - 1];
    goto LABEL_23;
  }
  v46 = v12;
  v19 = sub_30070B0((__int64)&v56);
  v17 = v10;
  if ( !v19 )
  {
    v59 = v46;
    v58 = 0;
    goto LABEL_13;
  }
  v35 = sub_3009970((__int64)&v56, v11, v46, v20, v21);
  v17 = v10;
  LOWORD(v15) = v35;
LABEL_23:
  v58 = v15;
  v59 = v32;
  if ( (_WORD)v15 )
    goto LABEL_6;
LABEL_13:
  v47 = v17;
  v22 = sub_3007260((__int64)&v58);
  v17 = v47;
  if ( v22 != 1 )
    return 0;
LABEL_14:
  if ( v8 == v11 && v9 == v17 || (unsigned __int8)sub_33E0780(v10, v54, 1, v14, a5, v17) )
  {
    v33 = sub_33FB960(a4, v52, v50);
    v28 = v56;
    v31 = v57;
    v30 = 187;
    *((_QWORD *)&v45 + 1) = v34;
    v29 = a3;
    *(_QWORD *)&v45 = v33;
    v44 = v51;
    return sub_3406EB0(a4, v30, v29, v28, v31, v27, v44, v45);
  }
  if ( (v9 != v49 || v8 != v48) && !(unsigned __int8)sub_33E0720(v52, v50, 1) )
  {
    if ( (unsigned __int8)sub_33E0780(v52, v50, 1, v23, v24, v25) )
    {
      *(_QWORD *)&v36 = sub_34015B0(a4, a3, v56, v57, 0, 0);
      *(_QWORD *)&v53 = sub_3406EB0(a4, 188, a3, v56, v57, v37, v51, v36);
      *((_QWORD *)&v53 + 1) = v38;
      *(_QWORD *)&v39 = sub_33FB960(a4, v10, v54);
      return sub_3406EB0(a4, 187, a3, v56, v57, DWORD2(v53), v53, v39);
    }
    if ( (unsigned __int8)sub_33E0720(v10, v54, 1) )
    {
      *(_QWORD *)&v40 = sub_34015B0(a4, a3, v56, v57, 0, 0);
      *(_QWORD *)&v55 = sub_3406EB0(a4, 188, a3, v56, v57, v41, v51, v40);
      *((_QWORD *)&v55 + 1) = v42;
      *(_QWORD *)&v43 = sub_33FB960(a4, v52, v50);
      v45 = v43;
      v44 = v55;
      goto LABEL_21;
    }
    return 0;
  }
  *(_QWORD *)&v26 = sub_33FB960(a4, v10, v54);
  v45 = v26;
  v44 = v51;
LABEL_21:
  v28 = v56;
  v29 = a3;
  v30 = 186;
  v31 = v57;
  return sub_3406EB0(a4, v30, v29, v28, v31, v27, v44, v45);
}
