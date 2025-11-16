// Function: sub_2A3F1A0
// Address: 0x2a3f1a0
//
void __fastcall sub_2A3F1A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // rax
  __int64 *v9; // r12
  __int64 v10; // r14
  __int64 *v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r15
  __int64 v17; // rbx
  __int64 *v18; // r14
  __int64 *v19; // r14
  __int64 v20; // rax
  __int64 *v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // r12
  __int64 v25; // r8
  _QWORD *v26; // rdx
  __int64 v27; // rdi
  _BYTE *v28; // rcx
  _QWORD *v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // r9
  __int64 *v32; // r8
  _QWORD *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 *v36; // rdx
  __int64 v37; // r12
  bool v38; // zf
  __int64 *v39; // r12
  __int64 v40; // rcx
  __int64 v41; // rsi
  _QWORD *v42; // rax
  _BYTE *v43; // rdx
  _BYTE **v44; // r12
  _BYTE **i; // r13
  _BYTE *v46; // rsi
  _QWORD *v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rax
  _BYTE **v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rsi
  __int64 v53; // rsi
  _QWORD *v54; // rax
  _BYTE *v55; // rdx
  _QWORD *v56; // rax
  _BYTE *v57; // rdx
  _QWORD *v58; // rax
  _BYTE *v59; // rdx
  __int64 *v61; // [rsp+8h] [rbp-398h]
  __int64 v62; // [rsp+10h] [rbp-390h] BYREF
  __int64 *v63; // [rsp+18h] [rbp-388h]
  __int64 v64; // [rsp+20h] [rbp-380h]
  int v65; // [rsp+28h] [rbp-378h]
  char v66; // [rsp+2Ch] [rbp-374h]
  char v67; // [rsp+30h] [rbp-370h] BYREF
  __int64 v68; // [rsp+130h] [rbp-270h] BYREF
  __int64 *v69; // [rsp+138h] [rbp-268h]
  __int64 v70; // [rsp+140h] [rbp-260h]
  int v71; // [rsp+148h] [rbp-258h]
  char v72; // [rsp+14Ch] [rbp-254h]
  _BYTE v73[256]; // [rsp+150h] [rbp-250h] BYREF
  __int64 v74; // [rsp+250h] [rbp-150h] BYREF
  _BYTE *v75; // [rsp+258h] [rbp-148h]
  __int64 v76; // [rsp+260h] [rbp-140h]
  int v77; // [rsp+268h] [rbp-138h]
  char v78; // [rsp+26Ch] [rbp-134h]
  _BYTE v79[304]; // [rsp+270h] [rbp-130h] BYREF

  v6 = *(__int64 **)a1;
  v7 = *(unsigned int *)(a1 + 8);
  v63 = (__int64 *)&v67;
  v8 = (__int64 *)v73;
  v9 = &v6[v7];
  v62 = 0;
  v64 = 32;
  v65 = 0;
  v66 = 1;
  v68 = 0;
  v69 = (__int64 *)v73;
  v70 = 32;
  v71 = 0;
  v72 = 1;
  if ( v6 == v9 )
  {
    v74 = 0;
    v75 = v79;
    v76 = 32;
    v77 = 0;
    v78 = 1;
    goto LABEL_129;
  }
  v10 = *v6;
LABEL_3:
  v11 = v63;
  a4 = HIDWORD(v64);
  v12 = &v63[HIDWORD(v64)];
  if ( v63 == v12 )
  {
LABEL_16:
    if ( HIDWORD(v64) < (unsigned int)v64 )
    {
      a4 = (unsigned int)++HIDWORD(v64);
      *v12 = v10;
      ++v62;
      goto LABEL_7;
    }
    goto LABEL_15;
  }
  while ( v10 != *v11 )
  {
    if ( v12 == ++v11 )
      goto LABEL_16;
  }
  while ( 1 )
  {
LABEL_7:
    v13 = *(_QWORD *)(v10 + 48);
    if ( !v13 )
      goto LABEL_13;
    if ( !v72 )
      goto LABEL_18;
    v14 = v69;
    a4 = HIDWORD(v70);
    v12 = &v69[HIDWORD(v70)];
    if ( v69 == v12 )
      break;
    while ( v13 != *v14 )
    {
      if ( v12 == ++v14 )
        goto LABEL_56;
    }
LABEL_13:
    if ( v9 == ++v6 )
      goto LABEL_19;
LABEL_14:
    v10 = *v6;
    if ( v66 )
      goto LABEL_3;
LABEL_15:
    sub_C8CC70((__int64)&v62, v10, (__int64)v12, a4, a5, a6);
  }
LABEL_56:
  if ( HIDWORD(v70) < (unsigned int)v70 )
  {
    a4 = (unsigned int)++HIDWORD(v70);
    *v12 = v13;
    ++v68;
    goto LABEL_13;
  }
LABEL_18:
  ++v6;
  sub_C8CC70((__int64)&v68, v13, (__int64)v12, a4, a5, a6);
  if ( v9 != v6 )
    goto LABEL_14;
LABEL_19:
  v74 = 0;
  v75 = v79;
  v8 = v69;
  v76 = 32;
  v77 = 0;
  v78 = 1;
  if ( !v72 )
  {
    v15 = (unsigned int)v70;
    v16 = &v69[(unsigned int)v70];
    goto LABEL_21;
  }
LABEL_129:
  v15 = HIDWORD(v70);
  v16 = &v8[HIDWORD(v70)];
LABEL_21:
  if ( v8 == v16 )
    goto LABEL_24;
  while ( 1 )
  {
    v17 = *v8;
    v18 = v8;
    if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v16 == ++v8 )
      goto LABEL_24;
  }
  if ( v16 != v8 )
  {
    v44 = *(_BYTE ***)(v17 + 24);
    if ( !*(_BYTE *)(v17 + 44) )
      goto LABEL_108;
LABEL_93:
    for ( i = &v44[*(unsigned int *)(v17 + 36)]; ; i = &v44[*(unsigned int *)(v17 + 32)] )
    {
      if ( v44 != i )
      {
        while ( 1 )
        {
          v46 = *v44;
          if ( (unsigned __int64)*v44 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( ++v44 == i )
            goto LABEL_97;
        }
        while ( v44 != i )
        {
          if ( *v46 )
            goto LABEL_122;
          if ( v66 )
          {
            v49 = v63;
            a4 = (__int64)&v63[HIDWORD(v64)];
            if ( v63 == (__int64 *)a4 )
              goto LABEL_122;
            while ( (_BYTE *)*v49 != v46 )
            {
              if ( (__int64 *)a4 == ++v49 )
                goto LABEL_122;
            }
          }
          else if ( !sub_C8CA60((__int64)&v62, (__int64)v46) )
          {
LABEL_122:
            if ( v44 != i )
              goto LABEL_102;
            break;
          }
          v50 = v44 + 1;
          if ( v44 + 1 == i )
            break;
          while ( 1 )
          {
            v46 = *v50;
            v44 = v50;
            if ( (unsigned __int64)*v50 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( i == ++v50 )
            {
              if ( v78 )
                goto LABEL_98;
              goto LABEL_120;
            }
          }
        }
      }
LABEL_97:
      if ( !v78 )
        break;
LABEL_98:
      v47 = v75;
      a4 = HIDWORD(v76);
      v15 = (__int64)&v75[8 * HIDWORD(v76)];
      if ( v75 == (_BYTE *)v15 )
      {
LABEL_124:
        if ( HIDWORD(v76) >= (unsigned int)v76 )
          break;
        a4 = (unsigned int)++HIDWORD(v76);
        *(_QWORD *)v15 = v17;
        ++v74;
      }
      else
      {
        while ( *v47 != v17 )
        {
          if ( (_QWORD *)v15 == ++v47 )
            goto LABEL_124;
        }
      }
LABEL_102:
      v48 = v18 + 1;
      if ( v18 + 1 == v16 )
        goto LABEL_24;
      v17 = *v48;
      for ( ++v18; (unsigned __int64)*v48 >= 0xFFFFFFFFFFFFFFFELL; v18 = v48 )
      {
        if ( v16 == ++v48 )
          goto LABEL_24;
        v17 = *v48;
      }
      if ( v16 == v18 )
        goto LABEL_24;
      v44 = *(_BYTE ***)(v17 + 24);
      if ( *(_BYTE *)(v17 + 44) )
        goto LABEL_93;
LABEL_108:
      ;
    }
LABEL_120:
    sub_C8CC70((__int64)&v74, v17, v15, a4, a5, a6);
    goto LABEL_102;
  }
LABEL_24:
  v19 = *(__int64 **)a1;
  v20 = 8LL * *(unsigned int *)(a1 + 8);
  v21 = (__int64 *)(*(_QWORD *)a1 + v20);
  v22 = v20 >> 3;
  v23 = v20 >> 5;
  if ( !v23 )
    goto LABEL_46;
  v24 = &v19[4 * v23];
  do
  {
    while ( 1 )
    {
      v25 = *(_QWORD *)(*v19 + 48);
      if ( v25 )
      {
        if ( v78 )
        {
          v26 = v75;
          v27 = HIDWORD(v76);
          v28 = &v75[8 * HIDWORD(v76)];
          v29 = v75;
          if ( v75 == v28 )
            goto LABEL_68;
          v30 = v75;
          while ( v25 != *v30 )
          {
            if ( v28 == (_BYTE *)++v30 )
              goto LABEL_68;
          }
          v31 = *(_QWORD *)(v19[1] + 48);
          if ( v31 )
          {
            v32 = v19 + 1;
            v33 = v75;
            goto LABEL_35;
          }
          v32 = v19 + 2;
          v34 = *(_QWORD *)(v19[2] + 48);
          if ( v34 )
            goto LABEL_82;
          goto LABEL_64;
        }
        if ( !sub_C8CA60((__int64)&v74, v25) )
          goto LABEL_68;
      }
      v31 = *(_QWORD *)(v19[1] + 48);
      if ( v31 )
      {
        v32 = v19 + 1;
        if ( v78 )
        {
          v26 = v75;
          v28 = &v75[8 * HIDWORD(v76)];
          v29 = v75;
          if ( v75 == v28 )
            goto LABEL_67;
          v33 = v75;
LABEL_35:
          while ( v31 != *v29 )
          {
            if ( ++v29 == (_QWORD *)v28 )
              goto LABEL_67;
          }
          v32 = v19 + 2;
          v34 = *(_QWORD *)(v19[2] + 48);
          if ( v34 )
            goto LABEL_39;
          goto LABEL_64;
        }
        v61 = v19 + 1;
        if ( !sub_C8CA60((__int64)&v74, v31) )
          goto LABEL_85;
      }
      v34 = *(_QWORD *)(v19[2] + 48);
      if ( v34 )
      {
        v32 = v19 + 2;
        if ( v78 )
        {
          v26 = v75;
          v27 = HIDWORD(v76);
LABEL_82:
          v28 = &v26[v27];
          v33 = v26;
          if ( v26 != (_QWORD *)v28 )
          {
LABEL_39:
            while ( v34 != *v33 )
            {
              if ( ++v33 == (_QWORD *)v28 )
                goto LABEL_67;
            }
            v32 = v19 + 3;
            v35 = *(_QWORD *)(v19[3] + 48);
            if ( v35 )
              goto LABEL_43;
            goto LABEL_44;
          }
LABEL_67:
          v19 = v32;
          goto LABEL_68;
        }
        v61 = v19 + 2;
        if ( !sub_C8CA60((__int64)&v74, v34) )
          goto LABEL_85;
      }
LABEL_64:
      v35 = *(_QWORD *)(v19[3] + 48);
      if ( !v35 )
        goto LABEL_44;
      v32 = v19 + 3;
      if ( !v78 )
        break;
      v26 = v75;
      v28 = &v75[8 * HIDWORD(v76)];
      if ( v28 == v75 )
        goto LABEL_67;
LABEL_43:
      while ( *v26 != v35 )
      {
        if ( v28 == (_BYTE *)++v26 )
          goto LABEL_67;
      }
LABEL_44:
      v19 += 4;
      if ( v19 == v24 )
        goto LABEL_45;
    }
    v61 = v19 + 3;
    if ( !sub_C8CA60((__int64)&v74, v35) )
    {
LABEL_85:
      v19 = v61;
LABEL_68:
      if ( v21 == v19 )
        goto LABEL_50;
      v39 = v19 + 1;
      if ( v21 == v19 + 1 )
        goto LABEL_50;
      while ( 1 )
      {
        v40 = *v39;
        v41 = *(_QWORD *)(*v39 + 48);
        if ( v41 )
        {
          if ( v78 )
          {
            v42 = v75;
            v43 = &v75[8 * HIDWORD(v76)];
            if ( v75 == v43 )
              goto LABEL_77;
            while ( v41 != *v42 )
            {
              if ( v43 == (_BYTE *)++v42 )
                goto LABEL_77;
            }
          }
          else
          {
            if ( !sub_C8CA60((__int64)&v74, v41) )
              goto LABEL_77;
            v40 = *v39;
          }
        }
        *v19++ = v40;
LABEL_77:
        if ( v21 == ++v39 )
          goto LABEL_50;
      }
    }
    v19 += 4;
  }
  while ( v19 != v24 );
LABEL_45:
  v22 = v21 - v19;
LABEL_46:
  if ( v22 == 2 )
    goto LABEL_135;
  if ( v22 == 3 )
  {
    v51 = *(_QWORD *)(*v19 + 48);
    if ( v51 )
    {
      if ( v78 )
      {
        v58 = v75;
        v59 = &v75[8 * HIDWORD(v76)];
        if ( v75 == v59 )
          goto LABEL_68;
        while ( v51 != *v58 )
        {
          if ( v59 == (_BYTE *)++v58 )
            goto LABEL_68;
        }
      }
      else if ( !sub_C8CA60((__int64)&v74, v51) )
      {
        goto LABEL_68;
      }
    }
    ++v19;
LABEL_135:
    v52 = *(_QWORD *)(*v19 + 48);
    if ( v52 )
    {
      if ( v78 )
      {
        v56 = v75;
        v57 = &v75[8 * HIDWORD(v76)];
        if ( v75 == v57 )
          goto LABEL_68;
        while ( v52 != *v56 )
        {
          if ( v57 == (_BYTE *)++v56 )
            goto LABEL_68;
        }
      }
      else if ( !sub_C8CA60((__int64)&v74, v52) )
      {
        goto LABEL_68;
      }
    }
    ++v19;
    goto LABEL_137;
  }
  if ( v22 != 1 )
    goto LABEL_49;
LABEL_137:
  v53 = *(_QWORD *)(*v19 + 48);
  if ( v53 )
  {
    if ( v78 )
    {
      v54 = v75;
      v55 = &v75[8 * HIDWORD(v76)];
      if ( v75 == v55 )
        goto LABEL_68;
      while ( v53 != *v54 )
      {
        if ( v55 == (_BYTE *)++v54 )
          goto LABEL_68;
      }
    }
    else if ( !sub_C8CA60((__int64)&v74, v53) )
    {
      goto LABEL_68;
    }
  }
LABEL_49:
  v19 = v21;
LABEL_50:
  v36 = *(__int64 **)a1;
  v37 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v21;
  if ( v21 != (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
  {
    memmove(v19, v21, *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v21);
    v36 = *(__int64 **)a1;
  }
  v38 = v78 == 0;
  *(_DWORD *)(a1 + 8) = ((char *)v19 + v37 - (char *)v36) >> 3;
  if ( v38 )
  {
    _libc_free((unsigned __int64)v75);
    if ( !v72 )
      goto LABEL_131;
LABEL_54:
    if ( !v66 )
LABEL_132:
      _libc_free((unsigned __int64)v63);
  }
  else
  {
    if ( v72 )
      goto LABEL_54;
LABEL_131:
    _libc_free((unsigned __int64)v69);
    if ( !v66 )
      goto LABEL_132;
  }
}
