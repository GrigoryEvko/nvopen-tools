// Function: sub_17D8620
// Address: 0x17d8620
//
unsigned __int64 __fastcall sub_17D8620(__int128 a1, int a2)
{
  __int64 *v2; // r14
  int v3; // eax
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  const char *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r13
  __int64 *v20; // r15
  __int64 **v21; // r14
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rbx
  unsigned __int64 *v29; // r12
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v38; // rax
  __m128i *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rbx
  _QWORD *v49; // rax
  __int64 *v50; // r15
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 *v53; // [rsp+8h] [rbp-108h]
  const char *v54; // [rsp+10h] [rbp-100h]
  __int64 v55; // [rsp+18h] [rbp-F8h]
  __int64 v56; // [rsp+30h] [rbp-E0h]
  __int64 v57; // [rsp+30h] [rbp-E0h]
  __int64 v59; // [rsp+38h] [rbp-D8h]
  __int64 v60; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v61[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v62; // [rsp+60h] [rbp-B0h]
  __m128i v63; // [rsp+70h] [rbp-A0h] BYREF
  __int64 *v64; // [rsp+80h] [rbp-90h]
  __int64 v65; // [rsp+90h] [rbp-80h] BYREF
  __int64 v66; // [rsp+98h] [rbp-78h]
  __int64 *v67; // [rsp+A0h] [rbp-70h]
  _QWORD *v68; // [rsp+A8h] [rbp-68h]

  v2 = (__int64 *)*((_QWORD *)&a1 + 1);
  sub_17CE510((__int64)&v65, *((__int64 *)&a1 + 1), 0, 0, 0);
  v3 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL);
  v4 = v3 & 0xFFFFFFF;
  if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) < 0 )
  {
    v5 = sub_1648A40(*((__int64 *)&a1 + 1));
    v7 = v5 + v6;
    if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
    {
      if ( (unsigned int)(v7 >> 4) )
        goto LABEL_47;
    }
    else if ( (unsigned int)((v7 - sub_1648A40(*((__int64 *)&a1 + 1))) >> 4) )
    {
      if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) < 0 )
      {
        v8 = *(_DWORD *)(sub_1648A40(*((__int64 *)&a1 + 1)) + 8);
        if ( *(char *)(*((_QWORD *)&a1 + 1) + 23LL) >= 0 )
          BUG();
        v9 = sub_1648A40(*((__int64 *)&a1 + 1));
        LODWORD(v10) = *(_DWORD *)(v9 + v10 - 4);
        v3 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL);
        if ( v4 - 2 != (_DWORD)v10 - v8 )
          goto LABEL_7;
        goto LABEL_11;
      }
LABEL_47:
      BUG();
    }
    v3 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL);
  }
  if ( v4 != 2 )
  {
LABEL_7:
    v11 = v3 & 0xFFFFFFF;
    v54 = *(const char **)(*((_QWORD *)&a1 + 1) - 24 * v11);
    v12 = *(const char **)(*((_QWORD *)&a1 + 1) + 24 * (1 - v11));
    goto LABEL_12;
  }
LABEL_11:
  v54 = 0;
  v12 = *(const char **)(*((_QWORD *)&a1 + 1) - 24LL * (v3 & 0xFFFFFFF));
LABEL_12:
  *((_QWORD *)&a1 + 1) = v12;
  v15 = (__int64)sub_17D4DA0(a1);
  if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
  {
    LOWORD(v64) = 257;
    v44 = sub_1643350(v68);
    v45 = sub_159C470(v44, 0, 0);
    v57 = sub_156D5F0(&v65, v15, v45, (__int64)&v63);
    if ( a2 == 1 )
    {
      v15 = v57;
    }
    else
    {
      v62 = 257;
      v46 = sub_1643350(v68);
      v47 = sub_159C470(v46, 1, 0);
      if ( *(_BYTE *)(v15 + 16) > 0x10u || *(_BYTE *)(v47 + 16) > 0x10u )
      {
        v55 = v47;
        LOWORD(v64) = 257;
        v49 = sub_1648A60(56, 2u);
        v48 = (__int64)v49;
        if ( v49 )
          sub_15FA320((__int64)v49, (_QWORD *)v15, v55, (__int64)&v63, 0);
        if ( v66 )
        {
          v50 = v67;
          sub_157E9D0(v66 + 40, v48);
          v51 = *(_QWORD *)(v48 + 24);
          v52 = *v50;
          *(_QWORD *)(v48 + 32) = v50;
          v52 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v48 + 24) = v52 | v51 & 7;
          *(_QWORD *)(v52 + 8) = v48 + 24;
          *v50 = *v50 & 7 | (v48 + 24);
        }
        sub_164B780(v48, v61);
        sub_12A86E0(&v65, v48);
      }
      else
      {
        v48 = sub_15A37D0((_BYTE *)v15, v47, 0);
      }
      LOWORD(v64) = 257;
      v15 = sub_156D390(&v65, v57, v48, (__int64)&v63);
    }
  }
  v16 = sub_17D4880(a1, v12, v13, v14);
  if ( !*(_BYTE *)(a1 + 488) )
  {
    *((_QWORD *)&a1 + 1) = v54;
    if ( v54 )
      goto LABEL_15;
LABEL_35:
    v40 = sub_17CDAE0((_QWORD *)a1, *v2);
    sub_17D4920(a1, v2, v40);
    v43 = sub_15A06D0(*(__int64 ***)(*(_QWORD *)(a1 + 8) + 184LL), (__int64)v2, v41, v42);
    sub_17D4B80(a1, (__int64)v2, v43);
    return sub_17CD270(&v65);
  }
  v63.m128i_i64[1] = v16;
  v38 = *(unsigned int *)(a1 + 504);
  v63.m128i_i64[0] = v15;
  v64 = v2;
  if ( (unsigned int)v38 >= *(_DWORD *)(a1 + 508) )
  {
    sub_16CD150(a1 + 496, (const void *)(a1 + 512), 0, 24, v17, v18);
    v38 = *(unsigned int *)(a1 + 504);
  }
  *((_QWORD *)&a1 + 1) = v54;
  v39 = (__m128i *)(*(_QWORD *)(a1 + 496) + 24 * v38);
  *v39 = _mm_loadu_si128(&v63);
  v39[1].m128i_i64[0] = (__int64)v64;
  ++*(_DWORD *)(a1 + 504);
  if ( !v54 )
    goto LABEL_35;
LABEL_15:
  v19 = 0;
  v53 = v2;
  v20 = sub_17D4DA0(a1);
  v56 = a2;
  v21 = **(__int64 ****)(*v20 + 16);
  do
  {
    while ( 1 )
    {
      v62 = 257;
      v22 = sub_1643350(v68);
      v23 = sub_159C470(v22, v19, 0);
      v26 = sub_15A06D0(v21, v19, v24, v25);
      if ( *((_BYTE *)v20 + 16) > 0x10u || *(_BYTE *)(v26 + 16) > 0x10u || *(_BYTE *)(v23 + 16) > 0x10u )
        break;
      ++v19;
      v20 = (__int64 *)sub_15A3890(v20, v26, v23, 0);
      if ( v56 == v19 )
        goto LABEL_30;
    }
    v59 = v26;
    LOWORD(v64) = 257;
    v27 = sub_1648A60(56, 3u);
    v28 = v27;
    if ( v27 )
      sub_15FA480((__int64)v27, v20, v59, v23, (__int64)&v63, 0);
    if ( v66 )
    {
      v29 = (unsigned __int64 *)v67;
      sub_157E9D0(v66 + 40, (__int64)v28);
      v30 = v28[3];
      v31 = *v29;
      v28[4] = (__int64)v29;
      v31 &= 0xFFFFFFFFFFFFFFF8LL;
      v28[3] = v31 | v30 & 7;
      *(_QWORD *)(v31 + 8) = v28 + 3;
      *v29 = *v29 & 7 | (unsigned __int64)(v28 + 3);
    }
    sub_164B780((__int64)v28, v61);
    if ( v65 )
    {
      v60 = v65;
      sub_1623A60((__int64)&v60, v65, 2);
      v32 = v28[6];
      if ( v32 )
        sub_161E7C0((__int64)(v28 + 6), v32);
      v33 = (unsigned __int8 *)v60;
      v28[6] = v60;
      if ( v33 )
        sub_1623210((__int64)&v60, v33, (__int64)(v28 + 6));
    }
    v20 = v28;
    ++v19;
  }
  while ( v56 != v19 );
LABEL_30:
  sub_17D4920(a1, v53, (__int64)v20);
  v36 = sub_17D4880(a1, v54, v34, v35);
  sub_17D4B80(a1, (__int64)v53, v36);
  return sub_17CD270(&v65);
}
