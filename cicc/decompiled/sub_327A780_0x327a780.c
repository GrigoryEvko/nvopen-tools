// Function: sub_327A780
// Address: 0x327a780
//
__int64 __fastcall sub_327A780(int a1, _QWORD *a2, int a3, __int64 a4)
{
  int v4; // r14d
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r12
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // edx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rax
  __int64 v19; // r12
  unsigned __int16 v20; // dx
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  int v32; // r9d
  __int64 v33; // rdx
  __int64 v34; // r13
  __int128 *v35; // rax
  int v36; // esi
  __int128 v37; // rax
  int v38; // r9d
  bool v39; // al
  __int64 v40; // rcx
  __int64 v41; // r8
  unsigned __int16 v42; // ax
  unsigned __int16 v43; // r9
  __int64 v44; // rdx
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // rdx
  __int128 v48; // [rsp-10h] [rbp-C0h]
  __int64 v49; // [rsp+8h] [rbp-A8h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+20h] [rbp-90h]
  unsigned int v52; // [rsp+20h] [rbp-90h]
  __int64 v53; // [rsp+20h] [rbp-90h]
  unsigned int v55; // [rsp+30h] [rbp-80h]
  __int128 v56; // [rsp+30h] [rbp-80h]
  int v57; // [rsp+40h] [rbp-70h] BYREF
  __int64 v58; // [rsp+48h] [rbp-68h]
  __int64 v59; // [rsp+50h] [rbp-60h]
  __int64 v60; // [rsp+58h] [rbp-58h]
  __int64 v61; // [rsp+60h] [rbp-50h] BYREF
  __int64 v62; // [rsp+68h] [rbp-48h]
  __int64 v63; // [rsp+70h] [rbp-40h]
  __int64 v64; // [rsp+78h] [rbp-38h]

  v4 = a4;
  if ( a1 == 56 )
  {
    v6 = a2[5];
    v7 = a2[6];
  }
  else
  {
    v6 = *a2;
    v7 = a2[1];
    a2 += 5;
  }
  v8 = *a2;
  v55 = *((_DWORD *)a2 + 2);
  if ( !(unsigned __int8)sub_33E2390(a4, v6, v7, 1) )
    return 0;
  if ( *(_DWORD *)(v8 + 24) != 192 )
    return 0;
  v10 = *(_QWORD **)(v8 + 40);
  v11 = *v10;
  v12 = v10[1];
  v13 = *((_DWORD *)v10 + 2);
  v14 = *(_QWORD *)(*v10 + 56LL);
  if ( !v14 )
    return 0;
  v15 = 1;
  do
  {
    if ( *(_DWORD *)(v14 + 8) == v13 )
    {
      if ( !v15 )
        return 0;
      v14 = *(_QWORD *)(v14 + 32);
      if ( !v14 )
        goto LABEL_16;
      if ( *(_DWORD *)(v14 + 8) == v13 )
        return 0;
      v15 = 0;
    }
    v14 = *(_QWORD *)(v14 + 32);
  }
  while ( v14 );
  if ( v15 == 1 )
    return 0;
LABEL_16:
  if ( !(unsigned __int8)sub_33DFCF0(v11, v12, 0) )
    return 0;
  v16 = *(_QWORD *)(v8 + 48) + 16LL * v55;
  v17 = *(_WORD *)v16;
  v58 = *(_QWORD *)(v16 + 8);
  v18 = *(_QWORD *)(v8 + 40);
  LOWORD(v57) = v17;
  v56 = (__int128)_mm_loadu_si128((const __m128i *)(v18 + 40));
  v19 = sub_33DFBC0(v56, *((_QWORD *)&v56 + 1), 0, 0);
  if ( !v19 )
    return 0;
  v20 = v57;
  v21 = v11;
  if ( (_WORD)v57 )
  {
    if ( (unsigned __int16)(v57 - 17) > 0xD3u )
    {
LABEL_20:
      v22 = v58;
      goto LABEL_21;
    }
    v20 = word_4456580[(unsigned __int16)v57 - 1];
    v22 = 0;
  }
  else
  {
    v39 = sub_30070B0((__int64)&v57);
    v20 = 0;
    v21 = v11;
    if ( !v39 )
      goto LABEL_20;
    v42 = sub_3009970((__int64)&v57, *((__int64 *)&v56 + 1), 0, v40, v41);
    v21 = v11;
    v43 = v42;
    v22 = v44;
    v20 = v43;
  }
LABEL_21:
  LOWORD(v61) = v20;
  v62 = v22;
  if ( v20 )
  {
    if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
      BUG();
    v27 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
  }
  else
  {
    v51 = v21;
    v23 = sub_3007260((__int64)&v61);
    v21 = v51;
    v24 = v23;
    v26 = v25;
    v59 = v24;
    v27 = v24;
    v60 = v26;
  }
  v28 = *(_QWORD *)(v19 + 96);
  v52 = *(_DWORD *)(v28 + 32);
  if ( v52 > 0x40 )
  {
    v49 = v21;
    v50 = v27;
    v45 = sub_C444A0(v28 + 24);
    v27 = v50;
    v21 = v49;
    if ( v52 - v45 > 0x40 )
      return 0;
    v29 = **(_QWORD **)(v28 + 24);
  }
  else
  {
    v29 = *(_QWORD *)(v28 + 24);
  }
  if ( v27 - 1 != v29 )
    return 0;
  v53 = v21;
  v61 = v6;
  v62 = v7;
  v63 = sub_3400BD0(v4, 1, a3, v57, v58, 0, 0);
  v64 = v30;
  if ( a1 == 56 )
  {
    v46 = sub_3402EA0(v4, 56, a3, v57, v58, 0, (__int64)&v61, 2);
    v34 = v47;
    v31 = v46;
    if ( v46 )
    {
      v35 = *(__int128 **)(v53 + 40);
      v36 = 191;
      goto LABEL_29;
    }
    return 0;
  }
  v31 = sub_3402EA0(v4, 57, a3, v57, v58, 0, (__int64)&v61, 2);
  v34 = v33;
  if ( !v31 )
    return 0;
  v35 = *(__int128 **)(v53 + 40);
  v36 = 192;
LABEL_29:
  *(_QWORD *)&v37 = sub_3406EB0(v4, v36, a3, v57, v58, v32, *v35, v56);
  *((_QWORD *)&v48 + 1) = v34;
  *(_QWORD *)&v48 = v31;
  return sub_3406EB0(v4, 56, a3, v57, v58, v38, v37, v48);
}
