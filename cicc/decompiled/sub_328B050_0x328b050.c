// Function: sub_328B050
// Address: 0x328b050
//
__int64 __fastcall sub_328B050(
        __int64 a1,
        _DWORD *a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        __int64 a7)
{
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r14
  int v13; // r15d
  unsigned __int16 *v15; // rdx
  unsigned __int16 v16; // ax
  __int64 v17; // r11
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // r15d
  unsigned int v24; // edx
  int v25; // r11d
  int v26; // r10d
  unsigned __int16 *v27; // rax
  unsigned int v28; // edx
  int v29; // r9d
  __int64 v30; // rbx
  __int64 v31; // rax
  __int128 v32; // [rsp-10h] [rbp-90h]
  int v33; // [rsp+0h] [rbp-80h]
  int v34; // [rsp+0h] [rbp-80h]
  int v35; // [rsp+8h] [rbp-78h]
  int v36; // [rsp+8h] [rbp-78h]
  int v37; // [rsp+8h] [rbp-78h]
  int v38; // [rsp+10h] [rbp-70h]
  __m128i v39; // [rsp+10h] [rbp-70h]
  __int128 v40; // [rsp+10h] [rbp-70h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  unsigned __int64 v42; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+30h] [rbp-50h]
  __int64 v44; // [rsp+40h] [rbp-40h] BYREF
  int v45; // [rsp+48h] [rbp-38h]

  v9 = sub_32719C0(a2, a3, a4, 0);
  if ( !v9 )
    return 0;
  v10 = v9;
  v11 = sub_32719C0(a2, a5, a6, 0);
  v12 = v11;
  if ( !v11 )
    return 0;
  v13 = *(_DWORD *)(v10 + 24);
  if ( v13 != *(_DWORD *)(v11 + 24) )
    return 0;
  if ( (v13 & 0xFFFFFFFD) != 0x4D )
    return 0;
  v15 = *(unsigned __int16 **)(a7 + 48);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  v18 = *(_QWORD *)(v10 + 48);
  if ( *(_WORD *)(v18 + 16) != v16 || *(_QWORD *)(v18 + 24) != v17 && !v16 )
    return 0;
  v19 = *(_QWORD *)(v12 + 48);
  if ( *(_WORD *)(v19 + 16) != v16 || *(_QWORD *)(v19 + 24) != v17 && !v16 )
    return 0;
  v35 = v16;
  v38 = v17;
  if ( (unsigned __int8)sub_33CFA90(v12, v10) )
  {
    v20 = v10;
    v10 = v12;
    v12 = v20;
  }
  v21 = *(_QWORD *)(v12 + 40);
  if ( *(_QWORD *)v21 == v10 && !*(_DWORD *)(v21 + 8) )
    goto LABEL_27;
  if ( *(_QWORD *)(v21 + 40) != v10 || *(_DWORD *)(v21 + 48) )
    return 0;
  if ( *(_QWORD *)v21 != v10 || *(_DWORD *)(v21 + 8) )
  {
    v22 = 0;
    if ( v13 == 79 )
      return 0;
  }
  else
  {
LABEL_27:
    v22 = 40;
  }
  v33 = v35;
  v36 = v38;
  v23 = (v13 != 77) + 72;
  v39 = _mm_loadu_si128((const __m128i *)(v21 + v22));
  if ( !(unsigned __int8)sub_328A020(
                           (__int64)a2,
                           v23,
                           **(_WORD **)(v10 + 48),
                           *(_QWORD *)(*(_QWORD *)(v10 + 48) + 8LL),
                           0) )
    return 0;
  v43 = sub_32719C0(a2, v39.m128i_i64[0], v39.m128i_u64[1], 1);
  v42 = v24 | v39.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  if ( !v43 )
    return 0;
  v25 = v36;
  v26 = v33;
  v44 = *(_QWORD *)(a7 + 80);
  if ( v44 )
  {
    sub_325F5D0(&v44);
    v26 = v33;
    v25 = v36;
  }
  v34 = v26;
  v37 = v25;
  v45 = *(_DWORD *)(a7 + 72);
  v27 = *(unsigned __int16 **)(v12 + 48);
  *((_QWORD *)&v32 + 1) = *((_QWORD *)v27 + 1);
  *(_QWORD *)&v32 = *v27;
  *(_QWORD *)&v40 = sub_33FB620(a1, v43, v42, (unsigned int)&v44, v27[8], *((_QWORD *)v27 + 3), v32);
  *((_QWORD *)&v40 + 1) = v28 | v42 & 0xFFFFFFFF00000000LL;
  v30 = sub_3412970(
          a1,
          v23,
          (unsigned int)&v44,
          *(_QWORD *)(v12 + 48),
          *(_DWORD *)(v12 + 68),
          v29,
          *(_OWORD *)*(_QWORD *)(v10 + 40),
          *(_OWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
          v40);
  sub_34161C0(a1, v12, 0, v30, 0);
  if ( *(_DWORD *)(a7 + 24) == 186 )
    v31 = sub_3400BD0(a1, 0, (unsigned int)&v44, v34, v37, 0, 0);
  else
    v31 = v30;
  v41 = v31;
  sub_9C6650(&v44);
  return v41;
}
