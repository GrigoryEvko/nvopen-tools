// Function: sub_32811A0
// Address: 0x32811a0
//
__int64 __fastcall sub_32811A0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 v7; // r10
  __int64 v8; // rdx
  __int64 v9; // r11
  bool v10; // zf
  unsigned int v11; // r15d
  unsigned int v12; // r14d
  __int64 v13; // r13
  char v15; // al
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int128 v18; // kr00_16
  int v19; // eax
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // r8
  const __m128i *v24; // rax
  unsigned __int16 *v25; // rax
  __int64 v26; // rdi
  int v27; // r9d
  unsigned __int16 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int16 v32; // r15
  unsigned __int16 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int16 v38; // ax
  __int64 v39; // rdx
  __int128 v40; // [rsp+20h] [rbp-90h]
  __int128 v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+30h] [rbp-80h]
  __int128 v43; // [rsp+30h] [rbp-80h]
  bool v44; // [rsp+40h] [rbp-70h]
  __int64 v45; // [rsp+40h] [rbp-70h]
  unsigned int v46; // [rsp+48h] [rbp-68h]
  __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  int v48; // [rsp+58h] [rbp-58h]
  unsigned __int64 v49; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v50; // [rsp+68h] [rbp-48h]
  __int16 v51; // [rsp+70h] [rbp-40h] BYREF
  __int64 v52; // [rsp+78h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v47 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v47, v4, 1);
  v48 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 **)(a2 + 40);
  v6 = v5[5];
  v7 = *v5;
  v8 = *v5;
  v9 = v5[1];
  v49 = 0;
  v10 = *(_DWORD *)(v6 + 24) == 58;
  v11 = *((_DWORD *)v5 + 2);
  v50 = 1;
  v12 = *((_DWORD *)v5 + 12);
  if ( !v10 )
  {
    v13 = 0;
    goto LABEL_5;
  }
  *(_QWORD *)&v40 = v7;
  *((_QWORD *)&v40 + 1) = v9;
  v42 = v8;
  v15 = sub_33D1410(v5[10], &v49);
  v16 = v50;
  v17 = v42;
  v18 = v40;
  if ( !v15 )
    goto LABEL_9;
  if ( v50 > 0x40 )
  {
    v46 = v50;
    v19 = sub_C444A0((__int64)&v49);
    v16 = v46;
    v17 = v42;
    v18 = v40;
    if ( v19 == v46 - 1 )
      goto LABEL_15;
LABEL_9:
    v13 = 0;
    goto LABEL_10;
  }
  if ( v49 != 1 )
    goto LABEL_9;
LABEL_15:
  v20 = *(_QWORD **)(v6 + 40);
  v21 = *(unsigned int *)(*v20 + 24LL);
  if ( (unsigned int)(v21 - 213) > 2 )
    goto LABEL_9;
  v22 = v20[5];
  v23 = *(unsigned int *)(v22 + 24);
  if ( (unsigned int)(v23 - 213) > 2 )
    goto LABEL_9;
  v24 = *(const __m128i **)(*v20 + 40LL);
  v43 = (__int128)_mm_loadu_si128(v24);
  v25 = (unsigned __int16 *)(*(_QWORD *)(v24->m128i_i64[0] + 48) + 16LL * v24->m128i_u32[2]);
  v41 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(v22 + 40));
  v26 = *(_QWORD *)(**(_QWORD **)(v22 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v22 + 40) + 8LL);
  v27 = *v25;
  if ( *(_WORD *)v26 != *v25 || *(_QWORD *)(v26 + 8) != *((_QWORD *)v25 + 1) && !*v25 )
    goto LABEL_9;
  if ( (_DWORD)v23 != (_DWORD)v21 )
    goto LABEL_9;
  v44 = *(_DWORD *)(a2 + 24) == 391;
  v28 = (unsigned __int16 *)(*(_QWORD *)(v17 + 48) + 16LL * v11);
  v29 = *v28;
  v30 = *((_QWORD *)v28 + 1);
  v51 = v29;
  v52 = v30;
  if ( (_WORD)v29 )
  {
    v31 = 0;
    v32 = word_4456580[(unsigned __int16)v29 - 1];
  }
  else
  {
    v38 = sub_3009970((__int64)&v51, v21, v29, v16, v23);
    v21 = (unsigned int)v21;
    v32 = v38;
    v31 = v39;
  }
  if ( ((_DWORD)v21 == 213) == v44
    || (v45 = v31,
        v33 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * v12),
        v34 = *v33,
        v35 = *((_QWORD *)v33 + 1),
        v51 = v34,
        v52 = v35,
        v32 == (unsigned __int16)sub_3281170(&v51, v21, v34, v31, v23))
    && (v32 || v45 == v36) )
  {
    v37 = sub_340F900(
            *a1,
            (unsigned int)((_DWORD)v21 != 213) + 391,
            (unsigned int)&v47,
            **(unsigned __int16 **)(a2 + 48),
            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
            v27,
            v18,
            v43,
            v41);
    LODWORD(v16) = v50;
    v13 = v37;
  }
  else
  {
    LODWORD(v16) = v50;
    v13 = 0;
  }
LABEL_10:
  if ( (unsigned int)v16 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
LABEL_5:
  if ( v47 )
    sub_B91220((__int64)&v47, v47);
  return v13;
}
