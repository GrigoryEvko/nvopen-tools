// Function: sub_1B19ED0
// Address: 0x1b19ed0
//
_QWORD *__fastcall sub_1B19ED0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        int a4,
        __int64 *a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v9; // r14
  _BYTE *v11; // rdi
  __int64 v12; // rbx
  unsigned int v13; // r12d
  __int64 v14; // rdx
  size_t v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 **v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rdi
  __int64 v25; // rsi
  __int64 *v26; // rbx
  __int64 *v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  _QWORD *v30; // r15
  unsigned __int8 v31; // al
  char v32; // dl
  _QWORD *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r15
  _QWORD *v37; // r12
  _QWORD *v39; // rax
  __int64 v40; // rdi
  unsigned __int64 *v41; // rbx
  __int64 v42; // rax
  unsigned __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // rdx
  unsigned __int8 *v47; // rsi
  char v48; // al
  _QWORD *v49; // rax
  __int64 v50; // rax
  unsigned __int64 *v51; // rbx
  __int64 v52; // rax
  unsigned __int64 v53; // rcx
  __int64 v54; // rsi
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  unsigned int v60; // [rsp+24h] [rbp-1ACh]
  __int64 v62; // [rsp+30h] [rbp-1A0h]
  __int64 v63; // [rsp+38h] [rbp-198h]
  unsigned __int8 *v64; // [rsp+48h] [rbp-188h] BYREF
  __int64 v65[2]; // [rsp+50h] [rbp-180h] BYREF
  __int16 v66; // [rsp+60h] [rbp-170h]
  __int64 v67[2]; // [rsp+70h] [rbp-160h] BYREF
  __int16 v68; // [rsp+80h] [rbp-150h]
  _BYTE *v69; // [rsp+90h] [rbp-140h] BYREF
  __int64 v70; // [rsp+98h] [rbp-138h]
  _BYTE s[304]; // [rsp+A0h] [rbp-130h] BYREF

  v9 = a2;
  v11 = s;
  v12 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  v13 = v12;
  v14 = (unsigned int)v12;
  v69 = s;
  v70 = 0x2000000000LL;
  if ( (unsigned int)v12 > 0x20 )
  {
    sub_16CD150((__int64)&v69, s, (unsigned int)v12, 8, (int)&v69, a6);
    v11 = v69;
    v14 = (unsigned int)v12;
  }
  v15 = 8 * v14;
  LODWORD(v70) = v12;
  if ( v15 )
    memset(v11, 0, v15);
  if ( (_DWORD)v12 != 1 )
  {
    v60 = a3 - 51;
    while ( 1 )
    {
      v13 >>= 1;
      if ( v13 )
      {
        v62 = v9;
        v16 = 0;
        v17 = a1;
        do
        {
          v18 = (__int64 *)&v69[8 * v16];
          v19 = sub_1643350(*(_QWORD **)(v17 + 24));
          v20 = v13 + (unsigned int)v16++;
          *v18 = sub_159C470(v19, v20, 0);
        }
        while ( v13 != v16 );
        a1 = v17;
        v9 = v62;
        v21 = 8LL * v13;
      }
      else
      {
        v21 = 0;
      }
      v22 = (__int64 **)sub_1643350(*(_QWORD **)(a1 + 24));
      v23 = sub_1599EF0(v22);
      v24 = (__int64 *)v69;
      v25 = (unsigned int)v70;
      v26 = (__int64 *)&v69[v21];
      v27 = (__int64 *)&v69[8 * (unsigned int)v70];
      if ( v26 != v27 )
      {
        do
          *v26++ = v23;
        while ( v27 != v26 );
        v24 = (__int64 *)v69;
        v25 = (unsigned int)v70;
      }
      v65[0] = (__int64)"rdx.shuf";
      v66 = 259;
      v28 = sub_15A01B0(v24, v25);
      v29 = sub_1599EF0(*(__int64 ***)v9);
      if ( *(_BYTE *)(v9 + 16) <= 0x10u && *(_BYTE *)(v29 + 16) <= 0x10u && *(_BYTE *)(v28 + 16) <= 0x10u )
        break;
      v63 = v29;
      v68 = 257;
      v39 = sub_1648A60(56, 3u);
      v30 = v39;
      if ( v39 )
        sub_15FA660((__int64)v39, (_QWORD *)v9, v63, (_QWORD *)v28, (__int64)v67, 0);
      v40 = *(_QWORD *)(a1 + 8);
      if ( v40 )
      {
        v41 = *(unsigned __int64 **)(a1 + 16);
        sub_157E9D0(v40 + 40, (__int64)v30);
        v42 = v30[3];
        v43 = *v41;
        v30[4] = v41;
        v43 &= 0xFFFFFFFFFFFFFFF8LL;
        v30[3] = v43 | v42 & 7;
        *(_QWORD *)(v43 + 8) = v30 + 3;
        *v41 = *v41 & 7 | (unsigned __int64)(v30 + 3);
      }
      sub_164B780((__int64)v30, v65);
      v44 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        goto LABEL_18;
      v64 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v64, v44, 2);
      v45 = v30[6];
      v46 = (__int64)(v30 + 6);
      if ( v45 )
      {
        sub_161E7C0((__int64)(v30 + 6), v45);
        v46 = (__int64)(v30 + 6);
      }
      v47 = v64;
      v30[6] = v64;
      if ( !v47 )
        goto LABEL_18;
      sub_1623210((__int64)&v64, v47, v46);
      if ( v60 > 1 )
      {
LABEL_19:
        v67[0] = (__int64)"bin.rdx";
        v68 = 259;
        v9 = sub_1904E90(a1, a3, v9, (__int64)v30, v67, 0, a7, a8, a9);
        v31 = *(_BYTE *)(v9 + 16);
        if ( v31 <= 0x17u )
        {
          if ( v31 == 5 )
          {
            v48 = *(_BYTE *)(*(_QWORD *)v9 + 8LL);
            if ( v48 == 16 )
              v48 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v9 + 16LL) + 8LL);
            if ( (unsigned __int8)(v48 - 1) <= 5u || *(_WORD *)(v9 + 18) == 52 )
LABEL_24:
              sub_15F2440(v9, -1);
          }
        }
        else
        {
          v32 = *(_BYTE *)(*(_QWORD *)v9 + 8LL);
          if ( v32 == 16 )
            v32 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v9 + 16LL) + 8LL);
          if ( (unsigned __int8)(v32 - 1) <= 5u || v31 == 76 )
            goto LABEL_24;
        }
        if ( a6 )
          goto LABEL_43;
        goto LABEL_26;
      }
LABEL_42:
      v9 = (__int64)sub_1B16290(a1, a4, (_BYTE *)v9, (__int64)v30);
      if ( a6 )
LABEL_43:
        sub_1B188F0((unsigned __int8 *)v9, a5, a6, 0);
LABEL_26:
      if ( v13 == 1 )
        goto LABEL_27;
    }
    v30 = (_QWORD *)sub_15A3950(v9, v29, (_BYTE *)v28, 0);
LABEL_18:
    if ( v60 > 1 )
      goto LABEL_19;
    goto LABEL_42;
  }
LABEL_27:
  v33 = *(_QWORD **)(a1 + 24);
  v66 = 257;
  v34 = sub_1643350(v33);
  v35 = sub_159C470(v34, 0, 0);
  v36 = v35;
  if ( *(_BYTE *)(v9 + 16) > 0x10u || *(_BYTE *)(v35 + 16) > 0x10u )
  {
    v68 = 257;
    v49 = sub_1648A60(56, 2u);
    v37 = v49;
    if ( v49 )
      sub_15FA320((__int64)v49, (_QWORD *)v9, v36, (__int64)v67, 0);
    v50 = *(_QWORD *)(a1 + 8);
    if ( v50 )
    {
      v51 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v50 + 40, (__int64)v37);
      v52 = v37[3];
      v53 = *v51;
      v37[4] = v51;
      v53 &= 0xFFFFFFFFFFFFFFF8LL;
      v37[3] = v53 | v52 & 7;
      *(_QWORD *)(v53 + 8) = v37 + 3;
      *v51 = *v51 & 7 | (unsigned __int64)(v37 + 3);
    }
    sub_164B780((__int64)v37, v65);
    v54 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v64 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v64, v54, 2);
      v55 = v37[6];
      if ( v55 )
        sub_161E7C0((__int64)(v37 + 6), v55);
      v56 = v64;
      v37[6] = v64;
      if ( v56 )
        sub_1623210((__int64)&v64, v56, (__int64)(v37 + 6));
    }
  }
  else
  {
    v37 = (_QWORD *)sub_15A37D0((_BYTE *)v9, v35, 0);
  }
  if ( v69 != s )
    _libc_free((unsigned __int64)v69);
  return v37;
}
