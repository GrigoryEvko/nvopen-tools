// Function: sub_3441EC0
// Address: 0x3441ec0
//
__int64 __fastcall sub_3441EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // r12
  int v11; // edx
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // r13d
  unsigned int v15; // esi
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 v18; // r10
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  char v22; // cl
  unsigned __int64 v23; // rax
  __int64 v24; // r9
  unsigned __int8 *v25; // r12
  __int64 v26; // rdx
  __int64 v27; // r13
  __int128 v28; // rax
  unsigned __int8 *v29; // rax
  unsigned __int8 *v30; // r10
  __int64 v31; // r11
  unsigned __int8 *v32; // r14
  __int64 v33; // rdx
  __int64 v34; // r15
  _QWORD *v35; // r12
  __int64 v36; // r13
  __int64 *v37; // rax
  __int64 v38; // rbx
  __int128 v39; // rax
  __int64 v40; // r9
  __int64 result; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  unsigned __int8 *v45; // rax
  __int64 v46; // rdx
  __int128 v47; // [rsp-30h] [rbp-D0h]
  __int128 v48; // [rsp-20h] [rbp-C0h]
  __int128 v49; // [rsp-10h] [rbp-B0h]
  __int64 v50; // [rsp+0h] [rbp-A0h]
  __int64 v51; // [rsp+8h] [rbp-98h]
  __int128 v52; // [rsp+10h] [rbp-90h]
  __int128 v53; // [rsp+20h] [rbp-80h]
  __int64 v54; // [rsp+20h] [rbp-80h]
  __int64 v55; // [rsp+20h] [rbp-80h]
  unsigned __int64 v56; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v57; // [rsp+38h] [rbp-68h]
  unsigned __int64 v58; // [rsp+40h] [rbp-60h] BYREF
  __int64 v59; // [rsp+48h] [rbp-58h]
  unsigned __int64 v60; // [rsp+50h] [rbp-50h] BYREF
  __int64 v61; // [rsp+58h] [rbp-48h]
  __int64 v62; // [rsp+60h] [rbp-40h]
  __int64 v63; // [rsp+68h] [rbp-38h]

  *((_QWORD *)&v53 + 1) = a3;
  *(_QWORD *)&v53 = a2;
  v8 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)a1 + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)a1 + 8LL));
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v58) = v9;
  v59 = v10;
  if ( !(_WORD)v9 )
  {
    if ( !sub_30070B0((__int64)&v58) )
    {
      v61 = v10;
      LOWORD(v60) = 0;
      goto LABEL_13;
    }
    LOWORD(v9) = sub_3009970((__int64)&v58, a2, v42, v43, v44);
LABEL_12:
    LOWORD(v60) = v9;
    v61 = v20;
    if ( (_WORD)v9 )
      goto LABEL_4;
LABEL_13:
    v12 = sub_3007260((__int64)&v60);
    v63 = v21;
    v13 = v12;
    v62 = v12;
    v14 = (unsigned int)v12 >> 1;
    v57 = v12;
    if ( (unsigned int)v12 > 0x40 )
      goto LABEL_7;
    goto LABEL_14;
  }
  if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
  {
    LOWORD(v9) = word_4456580[v9 - 1];
    v20 = 0;
    goto LABEL_12;
  }
  LOWORD(v60) = v9;
  v61 = v10;
LABEL_4:
  v11 = (unsigned __int16)v9;
  if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    BUG();
  v12 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v9 - 16];
  v57 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
  v13 = v12;
  v14 = (unsigned int)v12 >> 1;
  if ( (unsigned int)v12 > 0x40 )
  {
LABEL_7:
    sub_C43690((__int64)&v56, 0, 0);
    v13 = v57;
    v15 = v57 - v14;
    if ( v57 - v14 == v57 )
      goto LABEL_8;
    goto LABEL_15;
  }
LABEL_14:
  v56 = 0;
  v15 = v12 - v14;
  if ( (_DWORD)v12 == (_DWORD)v12 - v14 )
  {
    v16 = *(_QWORD *)(a1 + 8);
    v17 = *(_QWORD *)(a1 + 24);
    v18 = *(_QWORD *)(a1 + 16);
    goto LABEL_18;
  }
LABEL_15:
  if ( v15 <= 0x3F && v13 <= 0x40 )
  {
    v17 = *(_QWORD *)(a1 + 24);
    v18 = *(_QWORD *)(a1 + 16);
    v22 = 64 - v14;
    v16 = *(_QWORD *)(a1 + 8);
    v56 |= 0xFFFFFFFFFFFFFFFFLL >> v22 << v15;
    goto LABEL_18;
  }
  sub_C43C90(&v56, v15, v13);
  v13 = v57;
LABEL_8:
  v16 = *(_QWORD *)(a1 + 8);
  v17 = *(_QWORD *)(a1 + 24);
  LODWORD(v59) = v13;
  v18 = *(_QWORD *)(a1 + 16);
  if ( v13 <= 0x40 )
  {
LABEL_18:
    v23 = v56;
    goto LABEL_19;
  }
  v50 = v17;
  v51 = *(_QWORD *)(a1 + 16);
  sub_C43780((__int64)&v58, (const void **)&v56);
  v13 = v59;
  v18 = v51;
  v17 = v50;
  if ( (unsigned int)v59 > 0x40 )
  {
    sub_C43D10((__int64)&v58);
    v13 = v59;
    v19 = v58;
    v17 = v50;
    v18 = v51;
    goto LABEL_22;
  }
  v23 = v58;
LABEL_19:
  v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & ~v23;
  if ( !v13 )
    v19 = 0;
  v58 = v19;
LABEL_22:
  LODWORD(v61) = v13;
  v60 = v19;
  LODWORD(v59) = 0;
  v25 = sub_34007B0(v16, (__int64)&v60, v18, *(_DWORD *)v17, *(_QWORD *)(v17 + 8), 0, a6, 0);
  v27 = v26;
  if ( (unsigned int)v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( (unsigned int)v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  *((_QWORD *)&v49 + 1) = v27;
  *(_QWORD *)&v49 = v25;
  *((_QWORD *)&v48 + 1) = a5;
  *(_QWORD *)&v48 = a4;
  *(_QWORD *)&v28 = sub_3406EB0(
                      *(_QWORD **)(a1 + 8),
                      0xBAu,
                      *(_QWORD *)(a1 + 16),
                      **(unsigned int **)(a1 + 24),
                      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
                      v24,
                      v48,
                      v49);
  v29 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          186 - ((unsigned int)(**(_BYTE **)(a1 + 32) == 0) - 1),
          *(_QWORD *)(a1 + 16),
          **(unsigned int **)(a1 + 24),
          *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
          *(_QWORD *)(a1 + 16),
          v53,
          v28);
  v30 = v25;
  v31 = v27;
  v32 = v29;
  v34 = v33;
  if ( **(_BYTE **)(a1 + 32) )
  {
    v45 = sub_3400BD0(
            *(_QWORD *)(a1 + 8),
            0,
            *(_QWORD *)(a1 + 16),
            **(unsigned int **)(a1 + 24),
            *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
            0,
            a6,
            0);
    v31 = v46;
    v30 = v45;
  }
  v35 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v52 = v30;
  v36 = *(_QWORD *)(a1 + 16);
  *((_QWORD *)&v52 + 1) = v31;
  v37 = *(__int64 **)(a1 + 40);
  v54 = *v37;
  v38 = v37[1];
  *(_QWORD *)&v39 = sub_33ED040(v35, **(_DWORD **)(a1 + 48));
  *((_QWORD *)&v47 + 1) = v34;
  *(_QWORD *)&v47 = v32;
  result = sub_340F900(v35, 0xD0u, v36, v54, v38, v40, v47, v52, v39);
  if ( v57 > 0x40 )
  {
    if ( v56 )
    {
      v55 = result;
      j_j___libc_free_0_0(v56);
      return v55;
    }
  }
  return result;
}
