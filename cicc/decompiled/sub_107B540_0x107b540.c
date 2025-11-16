// Function: sub_107B540
// Address: 0x107b540
//
__int64 __fastcall sub_107B540(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        unsigned __int16 *a4,
        unsigned __int64 a5,
        _QWORD *a6)
{
  char *v8; // rbx
  unsigned __int64 v10; // r15
  __int64 v11; // r14
  char v12; // si
  char v13; // al
  char *v14; // rax
  unsigned __int64 v15; // r15
  __int64 v16; // r14
  char v17; // si
  char v18; // al
  char *v19; // rax
  unsigned __int64 v20; // r15
  __int64 v21; // r8
  void *v22; // rdi
  unsigned __int8 *v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // r15
  size_t v26; // r14
  size_t v27; // r13
  __int64 v28; // r15
  char v29; // si
  char v30; // al
  char *v31; // rax
  __int64 v32; // r13
  unsigned __int64 v33; // rdx
  char *v34; // rdi
  unsigned __int64 v35; // r14
  char v36; // si
  char v37; // al
  unsigned __int64 v38; // r14
  __int64 v39; // r15
  char v40; // si
  char v41; // al
  char *v42; // rax
  unsigned __int16 *v43; // rbx
  unsigned __int16 *v44; // r13
  unsigned __int64 v45; // r14
  __int64 v46; // r15
  char v47; // si
  char v48; // al
  char *v49; // rax
  unsigned __int64 v50; // r14
  __int64 v51; // r15
  char v52; // si
  char v53; // al
  char *v54; // rax
  _QWORD *i; // r13
  size_t v56; // r14
  size_t v57; // r15
  __int64 v58; // rbx
  char v59; // si
  char v60; // al
  char *v61; // rax
  __int64 v62; // r8
  unsigned __int64 v63; // rax
  _BYTE *v64; // rdi
  __int64 v65; // r14
  unsigned __int64 v66; // rbx
  char v67; // si
  char v68; // al
  char *v69; // rax
  unsigned int *v70; // rbx
  unsigned __int64 v71; // r14
  __int64 v72; // r15
  char v73; // si
  char v74; // al
  char *v75; // rax
  unsigned __int64 v76; // r15
  __int64 v77; // r14
  char v78; // si
  char v79; // al
  char *v80; // rax
  unsigned __int64 v81; // r15
  unsigned __int8 *v82; // r14
  __int64 v83; // r9
  void *v84; // rdi
  __int64 *v85; // rax
  __int64 *v86; // rax
  __int64 v90; // [rsp+28h] [rbp-88h]
  unsigned __int8 *src; // [rsp+30h] [rbp-80h]
  _QWORD *srcb; // [rsp+30h] [rbp-80h]
  void *srca; // [rsp+30h] [rbp-80h]
  _QWORD *srcc; // [rsp+30h] [rbp-80h]
  char *v95; // [rsp+38h] [rbp-78h]
  unsigned __int8 *v96; // [rsp+38h] [rbp-78h]
  unsigned __int8 *v97; // [rsp+38h] [rbp-78h]
  unsigned int *j; // [rsp+38h] [rbp-78h]
  __int64 v99[4]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v100[10]; // [rsp+60h] [rbp-50h] BYREF

  v8 = a2;
  sub_1079790(a1, (__int64)v99, (__int64)"linking", 7u);
  sub_107A5C0(2u, **(_QWORD **)(a1 + 104), 0);
  if ( a3 )
  {
    sub_1079610(a1, (__int64)v100, 8);
    sub_107A5C0(a3, **(_QWORD **)(a1 + 104), 0);
    v95 = &a2[120 * a3];
    if ( a2 != v95 )
    {
      do
      {
        v10 = (unsigned __int8)v8[16];
        v11 = **(_QWORD **)(a1 + 104);
        do
        {
          while ( 1 )
          {
            v12 = v10 & 0x7F;
            v13 = v10 & 0x7F | 0x80;
            v10 >>= 7;
            if ( v10 )
              v12 = v13;
            v14 = *(char **)(v11 + 32);
            if ( (unsigned __int64)v14 >= *(_QWORD *)(v11 + 24) )
              break;
            *(_QWORD *)(v11 + 32) = v14 + 1;
            *v14 = v12;
            if ( !v10 )
              goto LABEL_13;
          }
          sub_CB5D20(v11, v12);
        }
        while ( v10 );
LABEL_13:
        v15 = *((unsigned int *)v8 + 5);
        v16 = **(_QWORD **)(a1 + 104);
        do
        {
          while ( 1 )
          {
            v17 = v15 & 0x7F;
            v18 = v15 & 0x7F | 0x80;
            v15 >>= 7;
            if ( v15 )
              v17 = v18;
            v19 = *(char **)(v16 + 32);
            if ( (unsigned __int64)v19 >= *(_QWORD *)(v16 + 24) )
              break;
            *(_QWORD *)(v16 + 32) = v19 + 1;
            *v19 = v17;
            if ( !v15 )
              goto LABEL_19;
          }
          sub_CB5D20(v16, v17);
        }
        while ( v15 );
LABEL_19:
        switch ( v8[16] )
        {
          case 0:
          case 2:
          case 4:
          case 5:
            sub_107A5C0(*((unsigned int *)v8 + 24), **(_QWORD **)(a1 + 104), 0);
            if ( (*((_DWORD *)v8 + 5) & 0x50) != 0x10 )
            {
              v20 = *((_QWORD *)v8 + 1);
              src = *(unsigned __int8 **)v8;
              sub_107A5C0(v20, **(_QWORD **)(a1 + 104), 0);
              v21 = **(_QWORD **)(a1 + 104);
              v22 = *(void **)(v21 + 32);
              if ( v20 > *(_QWORD *)(v21 + 24) - (_QWORD)v22 )
              {
                sub_CB6200(**(_QWORD **)(a1 + 104), src, v20);
              }
              else if ( v20 )
              {
                v23 = src;
                srcb = **(_QWORD ***)(a1 + 104);
                memcpy(v22, v23, v20);
                srcb[4] += v20;
              }
            }
            break;
          case 1:
            v81 = *((_QWORD *)v8 + 1);
            v82 = *(unsigned __int8 **)v8;
            sub_107A5C0(v81, **(_QWORD **)(a1 + 104), 0);
            v83 = **(_QWORD **)(a1 + 104);
            v84 = *(void **)(v83 + 32);
            if ( v81 > *(_QWORD *)(v83 + 24) - (_QWORD)v84 )
            {
              sub_CB6200(**(_QWORD **)(a1 + 104), v82, v81);
            }
            else if ( v81 )
            {
              srcc = **(_QWORD ***)(a1 + 104);
              memcpy(v84, v82, v81);
              srcc[4] += v81;
            }
            if ( (v8[20] & 0x10) == 0 )
            {
              sub_107A5C0(*((unsigned int *)v8 + 24), **(_QWORD **)(a1 + 104), 0);
              sub_107A5C0(*((_QWORD *)v8 + 13), **(_QWORD **)(a1 + 104), 0);
              sub_107A5C0(*((_QWORD *)v8 + 14), **(_QWORD **)(a1 + 104), 0);
            }
            break;
          case 3:
            sub_107A5C0(
              *(unsigned int *)(*(_QWORD *)(a1 + 328) + 32LL * *((unsigned int *)v8 + 24) + 28),
              **(_QWORD **)(a1 + 104),
              0);
            break;
          default:
            BUG();
        }
        v8 += 120;
      }
      while ( v95 != v8 );
    }
    sub_1077B30(a1, v100);
    if ( !*(_DWORD *)(a1 + 744) )
      goto LABEL_3;
  }
  else if ( !*(_DWORD *)(a1 + 744) )
  {
LABEL_3:
    if ( a5 )
      goto LABEL_52;
LABEL_4:
    if ( a6[5] )
      goto LABEL_67;
    return sub_1077B30(a1, v99);
  }
  sub_1079610(a1, (__int64)v100, 5);
  sub_107A5C0(*(unsigned int *)(a1 + 744), **(_QWORD **)(a1 + 104), 0);
  v24 = *(_QWORD *)(a1 + 736);
  v25 = 80LL * *(unsigned int *)(a1 + 744);
  srca = (void *)(v24 + v25);
  if ( v24 == v24 + v25 )
    goto LABEL_51;
  do
  {
    v26 = *(_QWORD *)(v24 + 16);
    v96 = *(unsigned __int8 **)(v24 + 8);
    v27 = v26;
    v28 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v29 = v27 & 0x7F;
        v30 = v27 & 0x7F | 0x80;
        v27 >>= 7;
        if ( v27 )
          v29 = v30;
        v31 = *(char **)(v28 + 32);
        if ( (unsigned __int64)v31 >= *(_QWORD *)(v28 + 24) )
          break;
        *(_QWORD *)(v28 + 32) = v31 + 1;
        *v31 = v29;
        if ( !v27 )
          goto LABEL_34;
      }
      sub_CB5D20(v28, v29);
    }
    while ( v27 );
LABEL_34:
    v32 = **(_QWORD **)(a1 + 104);
    v33 = *(_QWORD *)(v32 + 24);
    v34 = *(char **)(v32 + 32);
    if ( v26 > v33 - (unsigned __int64)v34 )
    {
      sub_CB6200(**(_QWORD **)(a1 + 104), v96, v26);
      v85 = *(__int64 **)(a1 + 104);
      v32 = *v85;
      v34 = *(char **)(*v85 + 32);
      v33 = *(_QWORD *)(*v85 + 24);
    }
    else if ( v26 )
    {
      memcpy(v34, v96, v26);
      *(_QWORD *)(v32 + 32) += v26;
      v86 = *(__int64 **)(a1 + 104);
      v32 = *v86;
      v34 = *(char **)(*v86 + 32);
      v33 = *(_QWORD *)(*v86 + 24);
    }
    v35 = *(unsigned int *)(v24 + 40);
    while ( 1 )
    {
      v36 = v35 & 0x7F;
      v37 = v35 & 0x7F | 0x80;
      v35 >>= 7;
      if ( v35 )
        v36 = v37;
      if ( v33 <= (unsigned __int64)v34 )
        break;
      *(_QWORD *)(v32 + 32) = v34 + 1;
      *v34 = v36;
      if ( !v35 )
        goto LABEL_44;
LABEL_39:
      v34 = *(char **)(v32 + 32);
      v33 = *(_QWORD *)(v32 + 24);
    }
    sub_CB5D20(v32, v36);
    if ( v35 )
      goto LABEL_39;
LABEL_44:
    v38 = *(unsigned int *)(v24 + 44);
    v39 = **(_QWORD **)(a1 + 104);
    do
    {
      while ( 1 )
      {
        v40 = v38 & 0x7F;
        v41 = v38 & 0x7F | 0x80;
        v38 >>= 7;
        if ( v38 )
          v40 = v41;
        v42 = *(char **)(v39 + 32);
        if ( (unsigned __int64)v42 >= *(_QWORD *)(v39 + 24) )
          break;
        *(_QWORD *)(v39 + 32) = v42 + 1;
        *v42 = v40;
        if ( !v38 )
          goto LABEL_50;
      }
      sub_CB5D20(v39, v40);
    }
    while ( v38 );
LABEL_50:
    v24 += 80;
  }
  while ( srca != (void *)v24 );
LABEL_51:
  sub_1077B30(a1, v100);
  if ( !a5 )
    goto LABEL_4;
LABEL_52:
  sub_1079610(a1, (__int64)v100, 6);
  sub_107A5C0(a5, **(_QWORD **)(a1 + 104), 0);
  v43 = a4;
  v44 = &a4[4 * a5];
  if ( a4 != v44 )
  {
    do
    {
      v45 = *v43;
      v46 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v47 = v45 & 0x7F;
          v48 = v45 & 0x7F | 0x80;
          v45 >>= 7;
          if ( v45 )
            v47 = v48;
          v49 = *(char **)(v46 + 32);
          if ( (unsigned __int64)v49 >= *(_QWORD *)(v46 + 24) )
            break;
          *(_QWORD *)(v46 + 32) = v49 + 1;
          *v49 = v47;
          if ( !v45 )
            goto LABEL_59;
        }
        sub_CB5D20(v46, v47);
      }
      while ( v45 );
LABEL_59:
      v50 = *((unsigned int *)v43 + 1);
      v51 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v52 = v50 & 0x7F;
          v53 = v50 & 0x7F | 0x80;
          v50 >>= 7;
          if ( v50 )
            v52 = v53;
          v54 = *(char **)(v51 + 32);
          if ( (unsigned __int64)v54 >= *(_QWORD *)(v51 + 24) )
            break;
          *(_QWORD *)(v51 + 32) = v54 + 1;
          *v54 = v52;
          if ( !v50 )
            goto LABEL_65;
        }
        sub_CB5D20(v51, v52);
      }
      while ( v50 );
LABEL_65:
      v43 += 4;
    }
    while ( v44 != v43 );
  }
  sub_1077B30(a1, v100);
  if ( a6[5] )
  {
LABEL_67:
    sub_1079610(a1, (__int64)v100, 7);
    sub_107A5C0(a6[5], **(_QWORD **)(a1 + 104), 0);
    for ( i = (_QWORD *)a6[3]; a6 + 1 != i; i = (_QWORD *)sub_220EF30(i) )
    {
      v56 = i[5];
      v97 = (unsigned __int8 *)i[4];
      v57 = v56;
      v58 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v59 = v57 & 0x7F;
          v60 = v57 & 0x7F | 0x80;
          v57 >>= 7;
          if ( v57 )
            v59 = v60;
          v61 = *(char **)(v58 + 32);
          if ( (unsigned __int64)v61 >= *(_QWORD *)(v58 + 24) )
            break;
          *(_QWORD *)(v58 + 32) = v61 + 1;
          *v61 = v59;
          if ( !v57 )
            goto LABEL_74;
        }
        sub_CB5D20(v58, v59);
      }
      while ( v57 );
LABEL_74:
      v62 = **(_QWORD **)(a1 + 104);
      v63 = *(_QWORD *)(v62 + 24);
      v64 = *(_BYTE **)(v62 + 32);
      if ( v56 > v63 - (unsigned __int64)v64 )
      {
        sub_CB6200(**(_QWORD **)(a1 + 104), v97, v56);
        v62 = **(_QWORD **)(a1 + 104);
        v64 = *(_BYTE **)(v62 + 32);
        v63 = *(_QWORD *)(v62 + 24);
      }
      else if ( v56 )
      {
        v90 = **(_QWORD **)(a1 + 104);
        memcpy(v64, v97, v56);
        *(_QWORD *)(v90 + 32) += v56;
        v62 = **(_QWORD **)(a1 + 104);
        v64 = *(_BYTE **)(v62 + 32);
        v63 = *(_QWORD *)(v62 + 24);
      }
      if ( v63 <= (unsigned __int64)v64 )
      {
        sub_CB5D20(v62, 0);
      }
      else
      {
        *(_QWORD *)(v62 + 32) = v64 + 1;
        *v64 = 0;
      }
      v65 = **(_QWORD **)(a1 + 104);
      v66 = (__int64)(i[7] - i[6]) >> 3;
      do
      {
        while ( 1 )
        {
          v67 = v66 & 0x7F;
          v68 = v66 & 0x7F | 0x80;
          v66 >>= 7;
          if ( v66 )
            v67 = v68;
          v69 = *(char **)(v65 + 32);
          if ( (unsigned __int64)v69 >= *(_QWORD *)(v65 + 24) )
            break;
          *(_QWORD *)(v65 + 32) = v69 + 1;
          *v69 = v67;
          if ( !v66 )
            goto LABEL_85;
        }
        sub_CB5D20(v65, v67);
      }
      while ( v66 );
LABEL_85:
      v70 = (unsigned int *)i[6];
      for ( j = (unsigned int *)i[7]; j != v70; v70 += 2 )
      {
        v71 = *v70;
        v72 = **(_QWORD **)(a1 + 104);
        do
        {
          while ( 1 )
          {
            v73 = v71 & 0x7F;
            v74 = v71 & 0x7F | 0x80;
            v71 >>= 7;
            if ( v71 )
              v73 = v74;
            v75 = *(char **)(v72 + 32);
            if ( (unsigned __int64)v75 >= *(_QWORD *)(v72 + 24) )
              break;
            *(_QWORD *)(v72 + 32) = v75 + 1;
            *v75 = v73;
            if ( !v71 )
              goto LABEL_92;
          }
          sub_CB5D20(v72, v73);
        }
        while ( v71 );
LABEL_92:
        v76 = v70[1];
        v77 = **(_QWORD **)(a1 + 104);
        do
        {
          while ( 1 )
          {
            v78 = v76 & 0x7F;
            v79 = v76 & 0x7F | 0x80;
            v76 >>= 7;
            if ( v76 )
              v78 = v79;
            v80 = *(char **)(v77 + 32);
            if ( (unsigned __int64)v80 >= *(_QWORD *)(v77 + 24) )
              break;
            *(_QWORD *)(v77 + 32) = v80 + 1;
            *v80 = v78;
            if ( !v76 )
              goto LABEL_98;
          }
          sub_CB5D20(v77, v78);
        }
        while ( v76 );
LABEL_98:
        ;
      }
    }
    sub_1077B30(a1, v100);
  }
  return sub_1077B30(a1, v99);
}
