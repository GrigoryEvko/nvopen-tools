// Function: sub_23B5800
// Address: 0x23b5800
//
__int64 __fastcall sub_23B5800(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 *v11; // rdx
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rax
  unsigned __int64 *v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rdi
  __int64 v19; // r10
  __int64 *v20; // rdx
  __int64 *v21; // r12
  __int64 *v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 *v26; // rbx
  __int64 v27; // rdi
  int v28; // r10d
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int64 *v35; // rax
  __int64 v36; // rcx
  unsigned __int64 *v37; // r14
  unsigned __int64 v38; // rsi
  unsigned __int64 *v39; // rbx
  _BYTE *v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  _WORD *v44; // rdx
  unsigned __int64 *v45; // rax
  __int64 v46; // rcx
  unsigned __int64 *v47; // r14
  unsigned __int64 v48; // rsi
  unsigned __int64 *v49; // rbx
  _BYTE *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdx
  _WORD *v54; // rdx
  int v55; // edi
  __int64 v56; // rax
  __int64 v57; // rax
  int v58; // r11d

  if ( *(_BYTE *)(a2 + 32) && sub_23AED50(a2) )
    return sub_904010(a1, "Some blocks were deleted\n");
  v5 = *(_DWORD *)(a3 + 56);
  if ( *(_DWORD *)(a2 + 56) == v5 )
  {
    if ( !v5 )
      goto LABEL_4;
  }
  else
  {
    v7 = sub_904010(a1, "Different number of non-leaf basic blocks: before=");
    v8 = sub_CB59D0(v7, *(unsigned int *)(a2 + 56));
    v9 = sub_904010(v8, ", after=");
    v10 = sub_CB59D0(v9, *(unsigned int *)(a3 + 56));
    sub_904010(v10, "\n");
    if ( !*(_DWORD *)(a2 + 56) )
      goto LABEL_4;
  }
  v11 = *(unsigned __int64 **)(a2 + 48);
  v12 = &v11[5 * *(unsigned int *)(a2 + 64)];
  if ( v11 == v12 )
    goto LABEL_4;
  while ( 1 )
  {
    v13 = *v11;
    v14 = v11;
    if ( *v11 != -4096 && v13 != -8192 )
      break;
    v11 += 5;
    if ( v12 == v11 )
      goto LABEL_4;
  }
LABEL_12:
  if ( v12 == v14 )
  {
LABEL_4:
    result = *(unsigned int *)(a3 + 56);
    if ( !(_DWORD)result )
      return result;
    goto LABEL_21;
  }
  v15 = *(unsigned int *)(a3 + 64);
  v16 = *(_QWORD *)(a3 + 48);
  if ( (_DWORD)v15 )
  {
    v17 = (v15 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v18 = (__int64 *)(v16 + 40LL * v17);
    v19 = *v18;
    if ( *v18 == v13 )
    {
LABEL_15:
      if ( v18 != (__int64 *)(v16 + 40 * v15) )
        goto LABEL_16;
    }
    else
    {
      v55 = 1;
      while ( v19 != -4096 )
      {
        v58 = v55 + 1;
        v17 = (v15 - 1) & (v55 + v17);
        v18 = (__int64 *)(v16 + 40LL * v17);
        v19 = *v18;
        if ( *v18 == v13 )
          goto LABEL_15;
        v55 = v58;
      }
    }
  }
  sub_904010(a1, "Non-leaf block ");
  sub_23B2150(a1, *v14);
  v56 = sub_904010(a1, " is removed (");
  v57 = sub_CB59D0(v56, *((unsigned int *)v14 + 6));
  sub_904010(v57, " successors)\n");
LABEL_16:
  v14 += 5;
  if ( v14 == v12 )
    goto LABEL_4;
  do
  {
    v13 = *v14;
    if ( *v14 != -8192 && v13 != -4096 )
      goto LABEL_12;
    v14 += 5;
  }
  while ( v12 != v14 );
  result = *(unsigned int *)(a3 + 56);
  if ( (_DWORD)result )
  {
LABEL_21:
    v20 = *(__int64 **)(a3 + 48);
    result = 5LL * *(unsigned int *)(a3 + 64);
    v21 = &v20[5 * *(unsigned int *)(a3 + 64)];
    if ( v20 == v21 )
      return result;
    while ( 1 )
    {
      result = *v20;
      v22 = v20;
      if ( *v20 != -4096 && result != -8192 )
        break;
      v20 += 5;
      if ( v21 == v20 )
        return result;
    }
    while ( 1 )
    {
      if ( v21 == v22 )
        return result;
      v23 = *(unsigned int *)(a2 + 64);
      v24 = *(_QWORD *)(a2 + 48);
      if ( !(_DWORD)v23 )
        goto LABEL_40;
      v25 = (v23 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v26 = (__int64 *)(v24 + 40LL * v25);
      v27 = *v26;
      if ( result != *v26 )
        break;
LABEL_29:
      if ( v26 == (__int64 *)(v24 + 40 * v23) )
        goto LABEL_40;
      result = sub_23B56C0((__int64)(v26 + 1), (__int64)(v22 + 1));
      if ( !(_BYTE)result )
      {
        sub_904010(a1, "Different successors of block ");
        sub_23B2150(a1, *v22);
        sub_904010(a1, " (unordered):\n");
        v31 = sub_904010(a1, "- before (");
        v32 = sub_CB59D0(v31, *((unsigned int *)v26 + 6));
        sub_904010(v32, "): ");
        if ( *((_DWORD *)v26 + 6) )
        {
          v45 = (unsigned __int64 *)v26[2];
          v46 = 2LL * *((unsigned int *)v26 + 8);
          v47 = &v45[v46];
          if ( v45 != &v45[v46] )
          {
            while ( 1 )
            {
              v48 = *v45;
              v49 = v45;
              if ( *v45 != -8192 && v48 != -4096 )
                break;
              v45 += 2;
              if ( v47 == v45 )
                goto LABEL_42;
            }
            if ( v47 != v45 )
            {
              sub_23B2150(a1, v48);
              if ( *((_DWORD *)v49 + 2) != 1 )
              {
LABEL_74:
                v50 = *(_BYTE **)(a1 + 32);
                if ( *(_BYTE **)(a1 + 24) == v50 )
                {
                  v51 = sub_CB6200(a1, (unsigned __int8 *)"(", 1u);
                }
                else
                {
                  *v50 = 40;
                  v51 = a1;
                  ++*(_QWORD *)(a1 + 32);
                }
                v52 = sub_CB59D0(v51, *((unsigned int *)v49 + 2));
                v53 = *(_QWORD *)(v52 + 32);
                if ( (unsigned __int64)(*(_QWORD *)(v52 + 24) - v53) <= 2 )
                {
                  sub_CB6200(v52, "), ", 3u);
                }
                else
                {
                  *(_BYTE *)(v53 + 2) = 32;
                  *(_WORD *)v53 = 11305;
                  *(_QWORD *)(v52 + 32) += 3LL;
                }
                goto LABEL_78;
              }
              while ( 1 )
              {
                v54 = *(_WORD **)(a1 + 32);
                if ( *(_QWORD *)(a1 + 24) - (_QWORD)v54 <= 1u )
                {
                  sub_CB6200(a1, (unsigned __int8 *)", ", 2u);
                }
                else
                {
                  *v54 = 8236;
                  *(_QWORD *)(a1 + 32) += 2LL;
                }
LABEL_78:
                v49 += 2;
                if ( v49 == v47 )
                  break;
                while ( *v49 == -4096 || *v49 == -8192 )
                {
                  v49 += 2;
                  if ( v47 == v49 )
                    goto LABEL_42;
                }
                if ( v47 == v49 )
                  break;
                sub_23B2150(a1, *v49);
                if ( *((_DWORD *)v49 + 2) != 1 )
                  goto LABEL_74;
              }
            }
          }
        }
LABEL_42:
        sub_904010(a1, "\n");
        v33 = sub_904010(a1, "- after (");
        v34 = sub_CB59D0(v33, *((unsigned int *)v22 + 6));
        sub_904010(v34, "): ");
        if ( *((_DWORD *)v22 + 6) )
        {
          v35 = (unsigned __int64 *)v22[2];
          v36 = 2LL * *((unsigned int *)v22 + 8);
          v37 = &v35[v36];
          if ( v35 != &v35[v36] )
          {
            while ( 1 )
            {
              v38 = *v35;
              v39 = v35;
              if ( *v35 != -8192 && v38 != -4096 )
                break;
              v35 += 2;
              if ( v37 == v35 )
                goto LABEL_43;
            }
            if ( v37 != v35 )
            {
              sub_23B2150(a1, v38);
              if ( *((_DWORD *)v39 + 2) != 1 )
              {
LABEL_51:
                v40 = *(_BYTE **)(a1 + 32);
                if ( *(_BYTE **)(a1 + 24) == v40 )
                {
                  v41 = sub_CB6200(a1, (unsigned __int8 *)"(", 1u);
                }
                else
                {
                  *v40 = 40;
                  v41 = a1;
                  ++*(_QWORD *)(a1 + 32);
                }
                v42 = sub_CB59D0(v41, *((unsigned int *)v39 + 2));
                v43 = *(_QWORD *)(v42 + 32);
                if ( (unsigned __int64)(*(_QWORD *)(v42 + 24) - v43) <= 2 )
                {
                  sub_CB6200(v42, "), ", 3u);
                }
                else
                {
                  *(_BYTE *)(v43 + 2) = 32;
                  *(_WORD *)v43 = 11305;
                  *(_QWORD *)(v42 + 32) += 3LL;
                }
                goto LABEL_55;
              }
              while ( 1 )
              {
                v44 = *(_WORD **)(a1 + 32);
                if ( *(_QWORD *)(a1 + 24) - (_QWORD)v44 <= 1u )
                {
                  sub_CB6200(a1, (unsigned __int8 *)", ", 2u);
                }
                else
                {
                  *v44 = 8236;
                  *(_QWORD *)(a1 + 32) += 2LL;
                }
LABEL_55:
                v39 += 2;
                if ( v39 == v37 )
                  break;
                while ( *v39 == -8192 || *v39 == -4096 )
                {
                  v39 += 2;
                  if ( v37 == v39 )
                    goto LABEL_43;
                }
                if ( v37 == v39 )
                  break;
                sub_23B2150(a1, *v39);
                if ( *((_DWORD *)v39 + 2) != 1 )
                  goto LABEL_51;
              }
            }
          }
        }
LABEL_43:
        result = sub_904010(a1, "\n");
      }
LABEL_31:
      v22 += 5;
      if ( v22 == v21 )
        return result;
      while ( 1 )
      {
        result = *v22;
        if ( *v22 != -4096 && result != -8192 )
          break;
        v22 += 5;
        if ( v21 == v22 )
          return result;
      }
    }
    v28 = 1;
    while ( v27 != -4096 )
    {
      v25 = (v23 - 1) & (v28 + v25);
      v26 = (__int64 *)(v24 + 40LL * v25);
      v27 = *v26;
      if ( *v26 == result )
        goto LABEL_29;
      ++v28;
    }
LABEL_40:
    sub_904010(a1, "Non-leaf block ");
    sub_23B2150(a1, *v22);
    v29 = sub_904010(a1, " is added (");
    v30 = sub_CB59D0(v29, *((unsigned int *)v22 + 6));
    result = sub_904010(v30, " successors)\n");
    goto LABEL_31;
  }
  return result;
}
