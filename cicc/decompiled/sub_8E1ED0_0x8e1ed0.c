// Function: sub_8E1ED0
// Address: 0x8e1ed0
//
__m128i *__fastcall sub_8E1ED0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r12
  __int8 v5; // dl
  char v6; // di
  char v7; // si
  __int64 v8; // rcx
  char v9; // r8
  __m128i *result; // rax
  __int64 v11; // r8
  __m128i *v12; // r15
  __int64 i; // rbx
  __int8 v14; // al
  __int64 v15; // r14
  int v16; // eax
  int v17; // esi
  __m128i *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rbx
  __m128i *v21; // r14
  int v22; // edx
  __int8 v23; // al
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r14
  __int64 v27; // r15
  int v28; // r8d
  __int64 j; // rdi
  char v30; // r9
  __int64 v31; // rdi
  const __m128i *v32; // rbx
  __int64 v33; // r9
  __int8 v34; // cl
  __int64 k; // rdi
  char v36; // al
  __int64 m; // r10
  int v38; // eax
  __int64 v39; // r9
  int v40; // eax
  int v41; // esi
  __m128i *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // r15
  __int64 n; // r9
  __int64 v47; // rbx
  __int64 v48; // rcx
  const __m128i *v49; // r14
  __int64 v50; // r8
  int v51; // eax
  int v52; // eax
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r11
  __m128i *v56; // r10
  int v57; // eax
  __int64 v58; // r9
  const __m128i *v59; // r8
  int v60; // eax
  char v61; // dl
  char v62; // al
  _QWORD *v63; // rax
  int v64; // esi
  __m128i *v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  const __m128i *v68; // r9
  int v69; // esi
  int v70; // eax
  __m128i *v71; // rax
  _QWORD *v72; // rax
  int v73; // eax
  _QWORD *v74; // rax
  __m128i *v75; // r10
  __int64 v76; // [rsp+0h] [rbp-70h]
  __m128i *v77; // [rsp+0h] [rbp-70h]
  __int64 v78; // [rsp+8h] [rbp-68h]
  const __m128i *v79; // [rsp+8h] [rbp-68h]
  __int64 v80; // [rsp+10h] [rbp-60h]
  __m128i *v81; // [rsp+10h] [rbp-60h]
  __int64 v82; // [rsp+10h] [rbp-60h]
  const __m128i *v83; // [rsp+18h] [rbp-58h]
  __int64 v84; // [rsp+18h] [rbp-58h]
  __m128i *v85; // [rsp+20h] [rbp-50h]
  __m128i *v86; // [rsp+28h] [rbp-48h]
  const __m128i *v87; // [rsp+28h] [rbp-48h]
  __m128i *v88; // [rsp+28h] [rbp-48h]
  const __m128i *v89; // [rsp+28h] [rbp-48h]
  int v90[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v2 = a1;
  v4 = a2;
  v5 = *(_BYTE *)(a1 + 140);
  if ( v5 == 12 )
  {
    do
    {
      v2 = *(_QWORD *)(v2 + 160);
      v6 = *(_BYTE *)(v2 + 140);
    }
    while ( v6 == 12 );
  }
  else
  {
    v6 = *(_BYTE *)(a1 + 140);
  }
  v7 = *(_BYTE *)(a2 + 140);
  v8 = v4;
  if ( v7 == 12 )
  {
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
      v9 = *(_BYTE *)(v8 + 140);
    }
    while ( v9 == 12 );
  }
  else
  {
    v9 = v7;
  }
  if ( v6 == 6 )
  {
    if ( (*(_BYTE *)(v2 + 168) & 1) != 0 || v9 != 6 )
      goto LABEL_8;
    if ( (*(_BYTE *)(v8 + 168) & 1) == 0 )
    {
      v11 = *(_QWORD *)(v2 + 160);
      v12 = *(__m128i **)(v8 + 160);
      for ( i = v11; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v14 = v12[8].m128i_i8[12];
      if ( v14 == 12 )
      {
        v15 = *(_QWORD *)(v8 + 160);
        do
          v15 = *(_QWORD *)(v15 + 160);
        while ( *(_BYTE *)(v15 + 140) == 12 );
        if ( v15 == i )
          goto LABEL_103;
      }
      else
      {
        if ( v12 == (__m128i *)i )
        {
LABEL_26:
          if ( (v14 & 0xFB) != 8 )
          {
            v17 = 0;
LABEL_28:
            v18 = sub_73C570((const __m128i *)v11, v17);
            result = (__m128i *)sub_72D2E0(v18);
            goto LABEL_13;
          }
LABEL_103:
          v89 = (const __m128i *)v11;
          v60 = sub_8D4C10((__int64)v12, dword_4F077C4 != 2);
          v11 = (__int64)v89;
          v17 = v60;
          goto LABEL_28;
        }
        v15 = *(_QWORD *)(v8 + 160);
      }
      v86 = (__m128i *)v11;
      v16 = sub_8D97D0(i, v15, 0, v8, v11);
      v11 = (__int64)v86;
      if ( v16 )
      {
        v14 = v12[8].m128i_i8[12];
        goto LABEL_26;
      }
      v61 = *(_BYTE *)(i + 140);
      v62 = *(_BYTE *)(v15 + 140);
      if ( v61 != 1 )
      {
        if ( v62 == 1 )
        {
          if ( v61 == 7 )
            return 0;
          goto LABEL_111;
        }
        if ( v62 == 7 && v61 == 7 )
        {
          v73 = sub_8D97D0(i, v15, 0x2000u, v8, (__int64)v86);
          v11 = (__int64)v86;
          if ( !v73 )
            goto LABEL_30;
          if ( (unsigned int)sub_8D76D0(i) )
            result = (__m128i *)sub_72D2E0(v12);
          else
            result = (__m128i *)sub_72D2E0(v86);
          goto LABEL_13;
        }
        if ( (unsigned __int8)(v61 - 9) > 2u || (unsigned __int8)(v62 - 9) > 2u )
          goto LABEL_30;
        v63 = sub_8D5CE0(i, v15);
        v11 = (__int64)v86;
        if ( v63 )
        {
LABEL_111:
          v64 = 0;
          if ( (*(_BYTE *)(v11 + 140) & 0xFB) == 8 )
            v64 = sub_8D4C10(v11, dword_4F077C4 != 2);
          v65 = sub_73C570(v12, v64);
          result = (__m128i *)sub_72D2E0(v65);
          goto LABEL_13;
        }
        v19 = sub_8D5CE0(v15, i);
        v11 = (__int64)v86;
        if ( !v19 )
          goto LABEL_30;
LABEL_102:
        v17 = 0;
        if ( (v12[8].m128i_i8[12] & 0xFB) != 8 )
          goto LABEL_28;
        goto LABEL_103;
      }
      if ( v62 != 7 )
        goto LABEL_102;
    }
    return 0;
  }
  if ( v6 != 13 )
  {
    if ( v6 && v9 )
    {
      if ( v6 == 14 )
        return (__m128i *)a1;
      goto LABEL_9;
    }
    goto LABEL_12;
  }
  if ( v9 != 13 )
  {
LABEL_8:
    if ( v9 )
    {
LABEL_9:
      if ( v9 == 14 )
        return (__m128i *)v4;
      return 0;
    }
LABEL_12:
    result = (__m128i *)sub_72C930();
    goto LABEL_13;
  }
  v26 = *(_QWORD *)(v2 + 160);
  v27 = *(_QWORD *)(v8 + 160);
  v28 = *(unsigned __int8 *)(v26 + 140);
  for ( j = v26; (_BYTE)v28 == 12; v28 = *(unsigned __int8 *)(j + 140) )
    j = *(_QWORD *)(j + 160);
  v30 = *(_BYTE *)(v27 + 140);
  if ( v30 == 12 )
  {
    v31 = *(_QWORD *)(v8 + 160);
    do
    {
      v31 = *(_QWORD *)(v31 + 160);
      v30 = *(_BYTE *)(v31 + 140);
    }
    while ( v30 == 12 );
  }
  v11 = (unsigned int)(v28 - 9);
  if ( (unsigned __int8)v11 > 2u || (unsigned __int8)(v30 - 9) > 2u )
    goto LABEL_31;
  v32 = *(const __m128i **)(v2 + 168);
  v33 = *(_QWORD *)(v8 + 168);
  v34 = v32[8].m128i_i8[12];
  for ( k = (__int64)v32; v34 == 12; v34 = *(_BYTE *)(k + 140) )
    k = *(_QWORD *)(k + 160);
  v36 = *(_BYTE *)(v33 + 140);
  for ( m = v33; v36 == 12; v36 = *(_BYTE *)(m + 140) )
    m = *(_QWORD *)(m + 160);
  if ( v36 == 7 || v34 == 7 )
  {
    v43 = a1;
    if ( v5 == 12 )
    {
      do
        v43 = *(_QWORD *)(v43 + 160);
      while ( *(_BYTE *)(v43 + 140) == 12 );
    }
    v44 = v4;
    if ( v7 == 12 )
    {
      do
        v44 = *(_QWORD *)(v44 + 160);
      while ( *(_BYTE *)(v44 + 140) == 12 );
    }
    v45 = *(_QWORD *)(v44 + 160);
    for ( n = *(_QWORD *)(v43 + 160); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
    v47 = *(_QWORD *)(v44 + 160);
    if ( *(_BYTE *)(v45 + 140) == 12 )
    {
      do
        v47 = *(_QWORD *)(v47 + 160);
      while ( *(_BYTE *)(v47 + 140) == 12 );
    }
    v48 = *(_QWORD *)(v43 + 168);
    v85 = (__m128i *)v48;
    v49 = (const __m128i *)v48;
    v88 = *(__m128i **)(v44 + 168);
    if ( *(_BYTE *)(v48 + 140) == 12 )
    {
      do
        v49 = (const __m128i *)v49[10].m128i_i64[0];
      while ( v49[8].m128i_i8[12] == 12 );
    }
    v50 = *(_QWORD *)(v44 + 168);
    if ( v88[8].m128i_i8[12] == 12 )
    {
      do
        v50 = *(_QWORD *)(v50 + 160);
      while ( *(_BYTE *)(v50 + 140) == 12 );
    }
    else
    {
      v50 = *(_QWORD *)(v44 + 168);
    }
    v78 = n;
    v80 = *(_QWORD *)(v43 + 160);
    v83 = (const __m128i *)v50;
    v51 = sub_8DED30((__int64)v49, v50, 1048704, v48, v50);
    v11 = (__int64)v83;
    if ( !v51 || ((*(_BYTE *)(v83[10].m128i_i64[1] + 18) ^ *(_BYTE *)(v49[10].m128i_i64[1] + 18)) & 0x7F) != 0 )
      goto LABEL_30;
    v84 = v80;
    v76 = v11;
    v52 = sub_8D76D0((__int64)v49);
    v55 = v80;
    v56 = v85;
    if ( v52 )
      v56 = v88;
    if ( v80 != v45 )
    {
      v81 = v56;
      v57 = sub_8D97D0(v84, v45, 0, v53, v54);
      v55 = v84;
      v56 = v81;
      v58 = v78;
      v59 = (const __m128i *)v76;
      if ( !v57 )
      {
        v77 = v81;
        v79 = v59;
        v82 = v58;
        v72 = sub_8D5CE0(v84, v45);
        v55 = v84;
        v56 = v77;
        if ( !v72 )
        {
          v74 = sub_8D5CE0(v45, v84);
          v75 = v77;
          if ( !v74 )
            goto LABEL_30;
          if ( v85 == v77 )
          {
            v75 = sub_73F430(v49, 0);
            *(_QWORD *)(v75[10].m128i_i64[1] + 40) = v47;
            *(_BYTE *)(v75[10].m128i_i64[1] + 21) |= 1u;
          }
          result = (__m128i *)sub_73F0A0(v75, v45);
          goto LABEL_13;
        }
        if ( v88 == v77 )
        {
          v56 = sub_73F430(v79, 0);
          *(_QWORD *)(v56[10].m128i_i64[1] + 40) = v82;
          *(_BYTE *)(v56[10].m128i_i64[1] + 21) |= 1u;
          v55 = v84;
        }
      }
    }
    result = (__m128i *)sub_73F0A0(v56, v55);
    goto LABEL_13;
  }
  v87 = (const __m128i *)v33;
  v38 = sub_8E0BF0(k, m, 0, 1, v90);
  v39 = (__int64)v87;
  if ( !v38 )
    goto LABEL_30;
  if ( v26 == v27
    || (v40 = sub_8D97D0(v26, v27, 0, v8, v11), v39 = (__int64)v87, v40)
    || (v66 = sub_8D5CE0(v26, v27), v39 = (__int64)v87, v66) )
  {
    v41 = 0;
    if ( (*(_BYTE *)(v39 + 140) & 0xFB) == 8 )
      v41 = sub_8D4C10(v39, dword_4F077C4 != 2);
    v42 = sub_73C570(v32, v41);
    result = (__m128i *)sub_73F0A0(v42, v26);
  }
  else
  {
    v67 = sub_8D5CE0(v27, v26);
    v68 = v87;
    if ( !v67 )
      goto LABEL_30;
    v69 = 0;
    if ( (v32[8].m128i_i8[12] & 0xFB) == 8 )
    {
      v70 = sub_8D4C10((__int64)v32, dword_4F077C4 != 2);
      v68 = v87;
      v69 = v70;
    }
    v71 = sub_73C570(v68, v69);
    result = (__m128i *)sub_73F0A0(v71, v27);
  }
LABEL_13:
  if ( !result )
  {
LABEL_30:
    v5 = *(_BYTE *)(a1 + 140);
    v7 = *(_BYTE *)(v4 + 140);
LABEL_31:
    v20 = v4;
    v21 = (__m128i *)a1;
    while ( 1 )
    {
      while ( v5 == 12 )
      {
        v21 = (__m128i *)v21[10].m128i_i64[0];
        v5 = v21[8].m128i_i8[12];
      }
      for ( ; v7 == 12; v7 = *(_BYTE *)(v20 + 140) )
        v20 = *(_QWORD *)(v20 + 160);
      if ( v5 != v7 )
        return 0;
      if ( v5 == 7 )
      {
        if ( v21 == (__m128i *)v20 )
          return sub_8D4EC0(a1, v4, v21);
        v22 = 19;
      }
      else
      {
        if ( v21 == (__m128i *)v20 )
          return sub_8D4EC0(a1, v4, v21);
        v22 = 1;
      }
      if ( (unsigned int)sub_8DED30((__int64)v21, v20, v22, v8, v11) )
        return sub_8D4EC0(a1, v4, v21);
      v23 = v21[8].m128i_i8[12];
      if ( v23 == 8 )
        break;
      if ( v23 != 13 )
      {
        if ( v23 != 6 )
          return 0;
LABEL_46:
        v21 = (__m128i *)v21[10].m128i_i64[0];
        v20 = *(_QWORD *)(v20 + 160);
        goto LABEL_47;
      }
      v24 = v21[10].m128i_i64[0];
      v25 = *(_QWORD *)(v20 + 160);
      if ( v24 != v25 && !(unsigned int)sub_8D97D0(v24, v25, 0, v8, v11) )
        return 0;
      v21 = (__m128i *)v21[10].m128i_i64[1];
      v20 = *(_QWORD *)(v20 + 168);
LABEL_47:
      v5 = v21[8].m128i_i8[12];
      v7 = *(_BYTE *)(v20 + 140);
    }
    if ( !(unsigned int)sub_8D1590((__int64)v21, v20) )
    {
      if ( !dword_4D047EC )
        return 0;
      if ( (v21[10].m128i_i8[9] & 2) == 0 && (*(_BYTE *)(v20 + 169) & 2) == 0 )
        return 0;
    }
    goto LABEL_46;
  }
  return result;
}
