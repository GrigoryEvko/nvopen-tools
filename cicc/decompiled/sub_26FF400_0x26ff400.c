// Function: sub_26FF400
// Address: 0x26ff400
//
__int64 __fastcall sub_26FF400(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9)
{
  _BYTE *v9; // r15
  __int64 *v10; // rax
  __int64 v11; // r14
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v18; // r13
  __int64 *v19; // r12
  _QWORD *v20; // r10
  _QWORD *v21; // rcx
  char *v22; // r11
  char *v23; // rsi
  _QWORD *v24; // r10
  signed __int64 v25; // rdi
  char *v26; // r8
  char *v27; // rax
  char *v28; // rdx
  _QWORD *v29; // rax
  signed __int64 v30; // rdx
  __int64 v31; // r14
  __int64 *v32; // rax
  __int64 v33; // r15
  __m128i *v34; // r14
  unsigned __int64 v35; // r15
  __int64 *v36; // rsi
  unsigned __int64 v37; // r11
  _QWORD *v38; // rdx
  __int64 v39; // rax
  _QWORD *v40; // rdx
  signed __int64 v41; // rcx
  signed __int64 v42; // rax
  _DWORD *v43; // r10
  char v44; // al
  __int64 v45; // r14
  unsigned __int8 *v46; // r15
  char *v47; // rax
  __int64 v48; // rdx
  char *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r9
  __int64 *v52; // rax
  __int64 v53; // rdx
  __int64 *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rax
  unsigned __int64 v57; // r14
  char *v58; // rax
  __int64 v59; // rdx
  __int64 *v60; // rax
  __int64 *v61; // rax
  __int64 *v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rax
  unsigned __int64 v65; // r14
  char *v66; // rax
  __int64 v67; // rdx
  __int64 *v68; // rax
  __m128i *v69; // [rsp+50h] [rbp-90h]
  _QWORD *v70; // [rsp+50h] [rbp-90h]
  _QWORD *v71; // [rsp+50h] [rbp-90h]
  __int64 v72; // [rsp+60h] [rbp-80h]
  unsigned int v73; // [rsp+6Ch] [rbp-74h]
  __int64 *v74; // [rsp+70h] [rbp-70h]
  _DWORD *v75; // [rsp+70h] [rbp-70h]
  _DWORD *v76; // [rsp+70h] [rbp-70h]
  unsigned __int64 *v77; // [rsp+70h] [rbp-70h]
  unsigned __int64 *v78; // [rsp+70h] [rbp-70h]
  __int64 v79; // [rsp+80h] [rbp-60h]
  __int64 v84; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v85[7]; // [rsp+A8h] [rbp-38h] BYREF

  v9 = (_BYTE *)*a2;
  if ( *(_BYTE *)*a2 )
    return 0;
  v10 = *(__int64 **)(*((_QWORD *)v9 + 3) + 16LL);
  v11 = *v10;
  if ( *(_BYTE *)(*v10 + 8) != 12 )
    return 0;
  v73 = *(_DWORD *)(v11 + 8) >> 8;
  if ( *(_DWORD *)(v11 + 8) > 0x40FFu )
    return 0;
  v74 = &a2[4 * a3];
  if ( v74 != a2 )
  {
    v13 = a2 + 4;
    do
    {
      if ( sub_B2FC80((__int64)v9) )
        break;
      v14 = (*(__int64 (__fastcall **)(_QWORD, _BYTE *))(a1 + 8))(*(_QWORD *)(a1 + 16), v9);
      if ( (unsigned int)sub_25BFB50((__int64)v9, v14) || !*((_QWORD *)v9 + 13) )
        break;
      if ( (v9[2] & 1) != 0 )
        sub_B2C6D0((__int64)v9, v14, v15, v16);
      if ( *(_QWORD *)(*((_QWORD *)v9 + 12) + 16LL) || v11 != **(_QWORD **)(*((_QWORD *)v9 + 3) + 16LL) )
        break;
      if ( v74 == v13 )
        goto LABEL_16;
      v9 = (_BYTE *)*v13;
      v13 += 4;
    }
    while ( !*v9 );
    return 0;
  }
LABEL_16:
  v18 = *(_QWORD *)(a4 + 104);
  v72 = a4 + 88;
  if ( a4 + 88 == v18 )
    return 1;
  v19 = v74;
  do
  {
    if ( !(unsigned __int8)sub_26FDB00(
                             (__int64 *)a1,
                             (__int64)a2,
                             a3,
                             *(_QWORD *)(v18 + 32),
                             (__int64)(*(_QWORD *)(v18 + 40) - *(_QWORD *)(v18 + 32)) >> 3) )
      goto LABEL_61;
    v20 = 0;
    if ( !a5 )
      goto LABEL_41;
    v21 = (_QWORD *)a5[7];
    if ( !v21 )
    {
      v24 = a5 + 6;
      goto LABEL_39;
    }
    v22 = *(char **)(v18 + 40);
    v23 = *(char **)(v18 + 32);
    v24 = a5 + 6;
    v25 = v22 - v23;
    do
    {
      v26 = (char *)v21[5];
      v27 = (char *)v21[4];
      if ( v26 - v27 > v25 )
        v26 = &v27[v25];
      v28 = *(char **)(v18 + 32);
      if ( v27 != v26 )
      {
        while ( *(_QWORD *)v27 >= *(_QWORD *)v28 )
        {
          if ( *(_QWORD *)v27 > *(_QWORD *)v28 )
            goto LABEL_64;
          v27 += 8;
          v28 += 8;
          if ( v26 == v27 )
            goto LABEL_63;
        }
LABEL_29:
        v21 = (_QWORD *)v21[3];
        continue;
      }
LABEL_63:
      if ( v22 != v28 )
        goto LABEL_29;
LABEL_64:
      v24 = v21;
      v21 = (_QWORD *)v21[2];
    }
    while ( v21 );
    if ( v24 == a5 + 6 )
      goto LABEL_39;
    v29 = (_QWORD *)v24[4];
    v30 = v24[5] - (_QWORD)v29;
    if ( v25 > v30 )
      v22 = &v23[v30];
    if ( v23 == v22 )
    {
LABEL_75:
      if ( (_QWORD *)v24[5] != v29 )
        goto LABEL_39;
    }
    else
    {
      while ( *(_QWORD *)v23 >= *v29 )
      {
        if ( *(_QWORD *)v23 > *v29 )
          goto LABEL_40;
        v23 += 8;
        ++v29;
        if ( v22 == v23 )
          goto LABEL_75;
      }
LABEL_39:
      v85[0] = v18 + 32;
      v24 = (_QWORD *)sub_26FF1A0(a5 + 5, v24, v85);
    }
LABEL_40:
    v20 = v24 + 7;
LABEL_41:
    v31 = a2[2];
    v79 = v18 + 56;
    if ( v19 == a2 )
    {
LABEL_65:
      if ( *(_BYTE *)(v18 + 81) || *(_QWORD *)(v18 + 96) != *(_QWORD *)(v18 + 88) )
      {
        *(_DWORD *)v20 = 1;
        v20[1] = v31;
      }
      v49 = (char *)sub_BD5D20(*a2);
      sub_26F9AB0(a1, v79, v49, v50, v31, v51);
      if ( (*(_BYTE *)(a1 + 104) || (unsigned __int8)sub_C92250()) && v19 != a2 )
      {
        v52 = a2;
        do
        {
          *((_BYTE *)v52 + 25) = 1;
          v52 += 4;
        }
        while ( v19 != v52 );
      }
      goto LABEL_61;
    }
    v32 = a2 + 4;
    do
    {
      if ( v19 == v32 )
        goto LABEL_65;
      v32 += 4;
    }
    while ( v31 == *(v32 - 2) );
    v33 = (__int64)(*(_QWORD *)(v18 + 40) - *(_QWORD *)(v18 + 32)) >> 3;
    if ( v73 == 1 )
    {
      v53 = a2[2];
      v54 = a2;
      v55 = 0;
      while ( 1 )
      {
        if ( v53 == 1 )
        {
          if ( v55 )
          {
            v62 = a2;
            v63 = 0;
            while ( 1 )
            {
              if ( !v31 )
              {
                if ( v63 )
                  goto LABEL_46;
                v63 = v62[1];
              }
              v62 += 4;
              if ( v19 == v62 )
                break;
              v31 = v62[2];
            }
            v71 = v20;
            v78 = *(unsigned __int64 **)(v18 + 32);
            v64 = sub_26F6BC0(a1, v63);
            v65 = v64;
            if ( *(_BYTE *)(v18 + 81) || *(_QWORD *)(v18 + 96) != *(_QWORD *)(v18 + 88) )
            {
              *(_DWORD *)v71 = 2;
              v71[1] = 0;
              sub_26F8F40((__int64 *)a1, a8, a9, v78, v33, v64, "unique_member", 0xDu);
            }
            v66 = (char *)sub_BD5D20(*a2);
            sub_26FAF90(a1, v79, v66, v67, 0, v65);
            if ( *(_BYTE *)(a1 + 104) || (unsigned __int8)sub_C92250() )
            {
              v68 = a2;
              do
              {
                *((_BYTE *)v68 + 25) = 1;
                v68 += 4;
              }
              while ( v19 != v68 );
            }
            goto LABEL_61;
          }
          v55 = v54[1];
        }
        v54 += 4;
        if ( v19 == v54 )
        {
          v70 = v20;
          v77 = *(unsigned __int64 **)(v18 + 32);
          v56 = sub_26F6BC0(a1, v55);
          v57 = v56;
          if ( *(_BYTE *)(v18 + 81) || *(_QWORD *)(v18 + 96) != *(_QWORD *)(v18 + 88) )
          {
            *(_DWORD *)v70 = 2;
            v70[1] = 1;
            sub_26F8F40((__int64 *)a1, a8, a9, v77, v33, v56, "unique_member", 0xDu);
          }
          v58 = (char *)sub_BD5D20(*a2);
          sub_26FAF90(a1, v79, v58, v59, 1, v57);
          if ( *(_BYTE *)(a1 + 104) || (unsigned __int8)sub_C92250() )
          {
            v60 = a2;
            do
            {
              *((_BYTE *)v60 + 25) = 1;
              v60 += 4;
            }
            while ( v19 != v60 );
          }
          goto LABEL_61;
        }
        v53 = v54[2];
      }
    }
LABEL_46:
    v75 = v20;
    v34 = sub_26FE9C0((__int64)a2, a3, 0, v73);
    v35 = 0;
    v36 = a2;
    v69 = sub_26FE9C0((__int64)a2, a3, 1, v73);
    v37 = 0;
    do
    {
      v38 = (_QWORD *)v36[1];
      v39 = v38[1];
      v40 = (_QWORD *)*v38;
      v41 = (((unsigned __int64)v34->m128i_u64 + 7) >> 3) - 1 - v39 - (v40[3] - v40[2]);
      if ( v41 < 0 )
        v41 = 0;
      v37 += v41;
      v42 = (((unsigned __int64)v69->m128i_u64 + 7) >> 3) - 1 + v39 - v40[1] - (v40[9] - v40[8]);
      if ( v42 < 0 )
        v42 = 0;
      v36 += 4;
      v35 += v42;
    }
    while ( v19 != v36 );
    if ( v37 > v35 )
    {
      if ( v35 > 0x80 )
        goto LABEL_61;
      sub_26FD600((__int64)a2, a3, (unsigned __int64)v69, v73, (unsigned __int64 *)&v84, v85);
      v43 = v75;
    }
    else
    {
      if ( v37 > 0x80 )
        goto LABEL_61;
      sub_26FD280((__int64)a2, a3, (unsigned __int64)v34, v73, &v84, v85);
      v43 = v75;
    }
    if ( *(_BYTE *)(a1 + 104) || (v76 = v43, v44 = sub_C92250(), v43 = v76, v44) )
    {
      v61 = a2;
      do
      {
        *((_BYTE *)v61 + 25) = 1;
        v61 += 4;
      }
      while ( v19 != v61 );
    }
    if ( *(_BYTE *)(v18 + 81) || *(_QWORD *)(v18 + 96) != *(_QWORD *)(v18 + 88) )
    {
      *v43 = 3;
      sub_26F8FE0(
        (__int64 *)a1,
        a8,
        a9,
        *(unsigned __int64 **)(v18 + 32),
        (__int64)(*(_QWORD *)(v18 + 40) - *(_QWORD *)(v18 + 32)) >> 3,
        v84,
        a6,
        (unsigned int *)"byte");
      sub_26F8FE0(
        (__int64 *)a1,
        a8,
        a9,
        *(unsigned __int64 **)(v18 + 32),
        (__int64)(*(_QWORD *)(v18 + 40) - *(_QWORD *)(v18 + 32)) >> 3,
        1LL << SLOBYTE(v85[0]),
        a6,
        (unsigned int *)"bit");
    }
    v45 = sub_ACD640(*(_QWORD *)(a1 + 72), v84, 0);
    v46 = (unsigned __int8 *)sub_ACD640(*(_QWORD *)(a1 + 56), 1LL << SLOBYTE(v85[0]), 0);
    v47 = (char *)sub_BD5D20(*a2);
    sub_26FB610(a1, v79, v47, v48, v45, v46);
LABEL_61:
    v18 = sub_220EEE0(v18);
  }
  while ( v72 != v18 );
  return 1;
}
