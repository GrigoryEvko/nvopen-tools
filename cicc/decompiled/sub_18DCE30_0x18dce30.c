// Function: sub_18DCE30
// Address: 0x18dce30
//
void __fastcall sub_18DCE30(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  _QWORD *v9; // rcx
  unsigned int v10; // eax
  __int64 *v11; // rdx
  __int64 v12; // rdi
  _QWORD *v13; // rbx
  __int64 v14; // r12
  __int64 v15; // r12
  _QWORD *v16; // rax
  int v17; // r8d
  __int64 v18; // rbx
  char v19; // dl
  __int64 v20; // r13
  __int64 *v21; // rax
  __int64 *v22; // rsi
  unsigned int v23; // edi
  __int64 *v24; // rcx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 *v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *v30; // r13
  __int64 v31; // rdx
  _QWORD *v32; // r12
  _QWORD *v33; // rax
  __int64 *v34; // rax
  __int64 *v35; // rdi
  unsigned int v36; // r8d
  __int64 *v37; // rcx
  __int64 v38; // r14
  unsigned int v39; // r15d
  _QWORD *v40; // rax
  _QWORD *v41; // r9
  __int64 v42; // rax
  __int64 v43; // rsi
  _QWORD *v44; // rdx
  _QWORD *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rsi
  _QWORD *v48; // rax
  _QWORD *v49; // rsi
  unsigned int v50; // edi
  _QWORD *v51; // rcx
  _QWORD *v52; // rdx
  _QWORD *v53; // rsi
  unsigned int v54; // edi
  _QWORD *v55; // rcx
  _QWORD *v57; // [rsp+20h] [rbp-A0h]
  __int64 v59; // [rsp+28h] [rbp-98h]
  _QWORD *v61; // [rsp+38h] [rbp-88h]
  __int64 v62; // [rsp+38h] [rbp-88h]
  int v63; // [rsp+38h] [rbp-88h]
  _QWORD *v64; // [rsp+40h] [rbp-80h] BYREF
  __int64 v65; // [rsp+48h] [rbp-78h]
  _QWORD v66[14]; // [rsp+50h] [rbp-70h] BYREF

  v64 = v66;
  v66[1] = a4 + 24;
  v9 = v66;
  v66[0] = a3;
  v65 = 0x400000001LL;
  v10 = 1;
LABEL_2:
  v11 = &v9[2 * v10 - 2];
  v12 = *v11;
  v13 = (_QWORD *)v11[1];
  LODWORD(v65) = v10 - 1;
  v61 = *(_QWORD **)(v12 + 48);
  do
  {
    if ( v61 == v13 )
    {
      v15 = *(_QWORD *)(v12 + 8);
      if ( !v15 )
        goto LABEL_34;
      do
      {
        v16 = sub_1648700(v15);
        if ( (unsigned __int8)(*((_BYTE *)v16 + 16) - 25) <= 9u )
        {
          v62 = a2;
          v18 = a6;
          while ( 1 )
          {
            v20 = v16[5];
            v21 = *(__int64 **)(v18 + 8);
            if ( *(__int64 **)(v18 + 16) == v21 )
            {
              v22 = &v21[*(unsigned int *)(v18 + 28)];
              v23 = *(_DWORD *)(v18 + 28);
              if ( v21 != v22 )
              {
                v24 = 0;
                while ( v20 != *v21 )
                {
                  if ( *v21 == -2 )
                    v24 = v21;
                  if ( v22 == ++v21 )
                  {
                    if ( !v24 )
                      goto LABEL_48;
                    *v24 = v20;
                    --*(_DWORD *)(v18 + 32);
                    ++*(_QWORD *)v18;
                    goto LABEL_22;
                  }
                }
                goto LABEL_11;
              }
LABEL_48:
              if ( v23 < *(_DWORD *)(v18 + 24) )
                break;
            }
            sub_16CCBA0(v18, v20);
            if ( v19 )
            {
LABEL_22:
              v25 = (unsigned int)v65;
              v26 = v20 + 40;
              if ( (unsigned int)v65 < HIDWORD(v65) )
                goto LABEL_23;
LABEL_50:
              v59 = v26;
              sub_16CD150((__int64)&v64, v66, 0, 16, v17, v26);
              v25 = (unsigned int)v65;
              v26 = v59;
LABEL_23:
              v27 = &v64[2 * v25];
              *v27 = v20;
              v27[1] = v26;
              LODWORD(v65) = v65 + 1;
              v15 = *(_QWORD *)(v15 + 8);
              if ( !v15 )
              {
LABEL_24:
                v10 = v65;
                a6 = v18;
                a2 = v62;
                if ( !(_DWORD)v65 )
                  goto LABEL_25;
LABEL_37:
                v9 = v64;
                goto LABEL_2;
              }
              goto LABEL_12;
            }
            do
            {
LABEL_11:
              v15 = *(_QWORD *)(v15 + 8);
              if ( !v15 )
                goto LABEL_24;
LABEL_12:
              v16 = sub_1648700(v15);
            }
            while ( (unsigned __int8)(*((_BYTE *)v16 + 16) - 25) > 9u );
          }
          v26 = v20 + 40;
          *(_DWORD *)(v18 + 28) = v23 + 1;
          *v22 = v20;
          v25 = (unsigned int)v65;
          ++*(_QWORD *)v18;
          if ( (unsigned int)v25 >= HIDWORD(v65) )
            goto LABEL_50;
          goto LABEL_23;
        }
        v15 = *(_QWORD *)(v15 + 8);
      }
      while ( v15 );
LABEL_34:
      v33 = *(_QWORD **)(a5 + 8);
      if ( *(_QWORD **)(a5 + 16) != v33 )
      {
LABEL_35:
        sub_16CCBA0(a5, 0);
        goto LABEL_36;
      }
      v49 = &v33[*(unsigned int *)(a5 + 28)];
      v50 = *(_DWORD *)(a5 + 28);
      if ( v33 == v49 )
      {
LABEL_102:
        if ( v50 >= *(_DWORD *)(a5 + 24) )
          goto LABEL_35;
        *(_DWORD *)(a5 + 28) = v50 + 1;
        *v49 = 0;
        ++*(_QWORD *)a5;
      }
      else
      {
        v51 = 0;
        while ( *v33 )
        {
          if ( *v33 == -2 )
            v51 = v33;
          if ( v49 == ++v33 )
          {
            if ( !v51 )
              goto LABEL_102;
            *v51 = 0;
            --*(_DWORD *)(a5 + 32);
            ++*(_QWORD *)a5;
            goto LABEL_36;
          }
        }
      }
      goto LABEL_36;
    }
    v14 = 0;
    v13 = (_QWORD *)(*v13 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v13 )
      v14 = (__int64)(v13 - 3);
  }
  while ( !sub_18DCB50(a1, v14, a2, a7) );
  v34 = *(__int64 **)(a5 + 8);
  if ( *(__int64 **)(a5 + 16) != v34 )
    goto LABEL_39;
  v35 = &v34[*(unsigned int *)(a5 + 28)];
  v36 = *(_DWORD *)(a5 + 28);
  if ( v34 == v35 )
  {
LABEL_97:
    if ( v36 >= *(_DWORD *)(a5 + 24) )
    {
LABEL_39:
      sub_16CCBA0(a5, v14);
    }
    else
    {
      *(_DWORD *)(a5 + 28) = v36 + 1;
      *v35 = v14;
      ++*(_QWORD *)a5;
    }
  }
  else
  {
    v37 = 0;
    while ( v14 != *v34 )
    {
      if ( *v34 == -2 )
        v37 = v34;
      if ( v35 == ++v34 )
      {
        if ( !v37 )
          goto LABEL_97;
        *v37 = v14;
        --*(_DWORD *)(a5 + 32);
        ++*(_QWORD *)a5;
        break;
      }
    }
  }
LABEL_36:
  v10 = v65;
  if ( (_DWORD)v65 )
    goto LABEL_37;
LABEL_25:
  v28 = *(_QWORD **)(a6 + 16);
  if ( v28 == *(_QWORD **)(a6 + 8) )
    v29 = *(unsigned int *)(a6 + 28);
  else
    v29 = *(unsigned int *)(a6 + 24);
  v30 = &v28[v29];
  if ( v28 == v30 )
    goto LABEL_30;
  while ( 1 )
  {
    v31 = *v28;
    v32 = v28;
    if ( *v28 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v30 == ++v28 )
      goto LABEL_30;
  }
  while ( 1 )
  {
    if ( v30 == v32 )
      goto LABEL_30;
    if ( a3 != v31 && (*(_QWORD *)(v31 + 40) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v38 = (*(_QWORD *)(v31 + 40) & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v63 = sub_15F4D60(v38);
      if ( v63 )
        break;
    }
LABEL_77:
    v48 = v32 + 1;
    if ( v32 + 1 == v30 )
      goto LABEL_30;
    while ( 1 )
    {
      v31 = *v48;
      v32 = v48;
      if ( *v48 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v30 == ++v48 )
        goto LABEL_30;
    }
  }
  v39 = 0;
  while ( 2 )
  {
    v43 = sub_15F4DF0(v38, v39);
    if ( v43 == a3 )
      goto LABEL_60;
    v44 = *(_QWORD **)(a6 + 16);
    v40 = *(_QWORD **)(a6 + 8);
    if ( v44 == v40 )
    {
      v41 = &v40[*(unsigned int *)(a6 + 28)];
      if ( v40 == v41 )
      {
        v52 = *(_QWORD **)(a6 + 8);
      }
      else
      {
        do
        {
          if ( v43 == *v40 )
            break;
          ++v40;
        }
        while ( v41 != v40 );
        v52 = v41;
      }
LABEL_70:
      while ( v52 != v40 )
      {
        if ( *v40 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_59;
        ++v40;
      }
      if ( v40 == v41 )
        break;
      goto LABEL_60;
    }
    v57 = &v44[*(unsigned int *)(a6 + 24)];
    v40 = sub_16CC9F0(a6, v43);
    v41 = v57;
    if ( v43 == *v40 )
    {
      v46 = *(_QWORD *)(a6 + 16);
      if ( v46 == *(_QWORD *)(a6 + 8) )
        v47 = *(unsigned int *)(a6 + 28);
      else
        v47 = *(unsigned int *)(a6 + 24);
      v52 = (_QWORD *)(v46 + 8 * v47);
      goto LABEL_70;
    }
    v42 = *(_QWORD *)(a6 + 16);
    if ( v42 == *(_QWORD *)(a6 + 8) )
    {
      v40 = (_QWORD *)(v42 + 8LL * *(unsigned int *)(a6 + 28));
      v52 = v40;
      goto LABEL_70;
    }
    v40 = (_QWORD *)(v42 + 8LL * *(unsigned int *)(a6 + 24));
LABEL_59:
    if ( v40 != v41 )
    {
LABEL_60:
      if ( v63 == ++v39 )
        goto LABEL_77;
      continue;
    }
    break;
  }
  v45 = *(_QWORD **)(a5 + 8);
  if ( *(_QWORD **)(a5 + 16) != v45 )
    goto LABEL_73;
  v53 = &v45[*(unsigned int *)(a5 + 28)];
  v54 = *(_DWORD *)(a5 + 28);
  if ( v45 == v53 )
  {
LABEL_104:
    if ( v54 >= *(_DWORD *)(a5 + 24) )
    {
LABEL_73:
      sub_16CCBA0(a5, -1);
    }
    else
    {
      *(_DWORD *)(a5 + 28) = v54 + 1;
      *v53 = -1;
      ++*(_QWORD *)a5;
    }
  }
  else
  {
    v55 = 0;
    while ( *v45 != -1 )
    {
      if ( *v45 == -2 )
        v55 = v45;
      if ( v53 == ++v45 )
      {
        if ( !v55 )
          goto LABEL_104;
        *v55 = -1;
        --*(_DWORD *)(a5 + 32);
        ++*(_QWORD *)a5;
        break;
      }
    }
  }
LABEL_30:
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
}
