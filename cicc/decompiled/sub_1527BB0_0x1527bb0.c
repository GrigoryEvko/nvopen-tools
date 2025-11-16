// Function: sub_1527BB0
// Address: 0x1527bb0
//
void __fastcall sub_1527BB0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 *a5,
        __int64 a6,
        __int64 a7)
{
  int v7; // r14d
  _QWORD *v8; // r13
  __int64 v9; // r15
  unsigned int v10; // r12d
  unsigned __int8 *v11; // rbx
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  char v15; // dl
  unsigned __int8 *v16; // r14
  __int64 v17; // rbx
  unsigned __int8 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // eax
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  unsigned __int8 *v25; // rbx
  __int64 v26; // rdx
  unsigned __int64 v27; // rsi
  char v28; // al
  unsigned int v29; // r13d
  __int64 v30; // rdx
  unsigned __int64 v31; // rsi
  char v32; // al
  __int64 v33; // r14
  __int64 v34; // rax
  int *v35; // rbx
  int *v36; // r13
  __int64 v37; // r14
  int v38; // r12d
  __int64 v39; // rax
  __int64 v40; // r14
  unsigned int v41; // eax
  __int64 v42; // rbx
  int v43; // r14d
  __int64 v44; // rax
  __int64 v45; // rdi
  int v46; // r10d
  __int64 v47; // rdx
  int v48; // [rsp+4h] [rbp-7Ch]
  int v49; // [rsp+4h] [rbp-7Ch]
  int v50; // [rsp+4h] [rbp-7Ch]
  unsigned int v51; // [rsp+8h] [rbp-78h]
  unsigned int v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v55; // [rsp+20h] [rbp-60h]
  _QWORD *v59; // [rsp+40h] [rbp-40h]
  unsigned int v60; // [rsp+48h] [rbp-38h]
  int v61; // [rsp+4Ch] [rbp-34h]

  v7 = 0;
  v8 = (_QWORD *)a1;
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (a2 - 4));
  v60 = a6;
  v59 = (_QWORD *)v9;
  sub_1524D80((_DWORD *)a1, a2, *(_DWORD *)(a1 + 16));
  v61 = *(_DWORD *)(v9 + 8);
  if ( *(_BYTE *)(a7 + 4) )
  {
    v7 = 1;
    if ( (*(_BYTE *)(*(_QWORD *)v9 + 8LL) & 1) == 0 )
      sub_1527B10((_DWORD *)a1, *(_QWORD *)v9, *(_DWORD *)a7);
  }
  v10 = 0;
  v11 = a5;
  v55 = &a5[a6];
  if ( v7 != v61 )
  {
    LODWORD(v12) = v7;
    while ( 1 )
    {
      v13 = *v59 + 16LL * (unsigned int)v12;
      v14 = *(_BYTE *)(v13 + 8);
      if ( (v14 & 1) != 0 )
        goto LABEL_30;
      v15 = (v14 >> 1) & 7;
      if ( v15 == 3 )
        break;
      if ( v15 != 5 )
      {
        v22 = *(unsigned int *)(a3 + 4LL * v10);
        if ( v15 == 2 )
        {
          if ( *(_QWORD *)v13 )
          {
            ++v10;
            sub_1525280(v8, v22, *(_QWORD *)v13);
            goto LABEL_22;
          }
        }
        else
        {
          if ( v15 == 4 )
          {
            if ( (unsigned __int8)(v22 - 97) <= 0x19u )
            {
              LODWORD(v22) = (char)v22 - 97;
            }
            else if ( (unsigned __int8)(v22 - 65) <= 0x19u )
            {
              LODWORD(v22) = (char)v22 - 39;
            }
            else if ( (unsigned __int8)(v22 - 48) <= 9u )
            {
              LODWORD(v22) = (char)v22 + 4;
            }
            else
            {
              LODWORD(v22) = ((_BYTE)v22 != 46) + 62;
            }
            LODWORD(v23) = 6;
          }
          else
          {
            v23 = *(_QWORD *)v13;
            if ( !*(_QWORD *)v13 )
            {
              ++v10;
              goto LABEL_22;
            }
          }
          sub_1524D80(v8, v22, v23);
        }
LABEL_30:
        ++v10;
        goto LABEL_22;
      }
      if ( !v11 )
      {
        v33 = a3 + 4LL * v10;
        sub_1524E40(v8, a4 - v10, 6);
        v34 = a4 - v10;
        if ( *((_DWORD *)v8 + 2) )
        {
          v45 = *v8;
          v46 = *((_DWORD *)v8 + 3);
          v47 = *(unsigned int *)(*v8 + 8LL);
          if ( (unsigned __int64)*(unsigned int *)(*v8 + 12LL) - v47 <= 3 )
          {
            v50 = *((_DWORD *)v8 + 3);
            sub_16CD150(v45, v45 + 16, v47 + 4, 1);
            v46 = v50;
            v34 = a4 - v10;
            v47 = *(unsigned int *)(v45 + 8);
          }
          *(_DWORD *)(*(_QWORD *)v45 + v47) = v46;
          *(_DWORD *)(v45 + 8) += 4;
          v8[1] = 0;
        }
        if ( v33 != v33 + 4 * v34 )
        {
          v51 = v10;
          v35 = (int *)(a3 + 4LL * v10);
          v49 = v12;
          v12 = (unsigned __int64)v8;
          v36 = (int *)(v33 + 4 * v34);
          do
          {
            v37 = *(_QWORD *)v12;
            v38 = *v35;
            v39 = *(unsigned int *)(*(_QWORD *)v12 + 8LL);
            if ( (unsigned int)v39 >= *(_DWORD *)(*(_QWORD *)v12 + 12LL) )
            {
              sub_16CD150(*(_QWORD *)v12, v37 + 16, 0, 1);
              v39 = *(unsigned int *)(v37 + 8);
            }
            ++v35;
            *(_BYTE *)(*(_QWORD *)v37 + v39) = v38;
            ++*(_DWORD *)(v37 + 8);
          }
          while ( v36 != v35 );
          v8 = (_QWORD *)v12;
          v11 = 0;
          v10 = v51;
          LODWORD(v12) = v49;
        }
        while ( 1 )
        {
          v40 = *v8;
          v41 = *(_DWORD *)(*v8 + 8LL);
          if ( (v41 & 3) == 0 )
            break;
          if ( v41 >= *(_DWORD *)(v40 + 12) )
            sub_16CD150(*v8, v40 + 16, 0, 1);
          *(_BYTE *)(*(_QWORD *)v40 + (unsigned int)(*(_DWORD *)(v40 + 8))++) = 0;
        }
        goto LABEL_22;
      }
      sub_1524E40(v8, v60, 6);
      if ( *((_DWORD *)v8 + 2) )
      {
        v42 = *v8;
        v43 = *((_DWORD *)v8 + 3);
        v44 = *(unsigned int *)(*v8 + 8LL);
        if ( (unsigned __int64)*(unsigned int *)(*v8 + 12LL) - v44 <= 3 )
        {
          sub_16CD150(*v8, v42 + 16, v44 + 4, 1);
          v44 = *(unsigned int *)(v42 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v42 + v44) = v43;
        *(_DWORD *)(v42 + 8) += 4;
        v8[1] = 0;
      }
      v16 = a5;
      if ( a5 != v55 )
      {
        v52 = v10;
        do
        {
          v17 = *v8;
          v18 = *v16;
          v19 = *(unsigned int *)(*v8 + 8LL);
          if ( (unsigned int)v19 >= *(_DWORD *)(*v8 + 12LL) )
          {
            sub_16CD150(*v8, v17 + 16, 0, 1);
            v19 = *(unsigned int *)(v17 + 8);
          }
          ++v16;
          *(_BYTE *)(*(_QWORD *)v17 + v19) = v18;
          ++*(_DWORD *)(v17 + 8);
        }
        while ( v55 != v16 );
        v20 = *v8;
        v10 = v52;
        v21 = *(_DWORD *)(*v8 + 8LL);
        if ( (v21 & 3) == 0 )
          goto LABEL_21;
        goto LABEL_17;
      }
      while ( 1 )
      {
        v20 = *v8;
        v21 = *(_DWORD *)(*v8 + 8LL);
        if ( (v21 & 3) == 0 )
          break;
LABEL_17:
        if ( *(_DWORD *)(v20 + 12) <= v21 )
          sub_16CD150(v20, v20 + 16, 0, 1);
        *(_BYTE *)(*(_QWORD *)v20 + (unsigned int)(*(_DWORD *)(v20 + 8))++) = 0;
      }
LABEL_21:
      v11 = 0;
LABEL_22:
      LODWORD(v12) = v12 + 1;
      if ( v61 == (_DWORD)v12 )
        return;
    }
    v12 = (unsigned int)(v12 + 1);
    v53 = *v59 + 16 * v12;
    if ( v11 )
    {
      sub_1524E40(v8, v60, 6);
      if ( !v60 )
        goto LABEL_21;
      v24 = v11;
      v25 = &v11[(unsigned int)a6];
      while ( 1 )
      {
        v27 = *v24;
        v28 = (*(_BYTE *)(v53 + 8) >> 1) & 7;
        if ( v28 == 2 )
          break;
        if ( v28 == 4 )
        {
          if ( (unsigned __int8)(v27 - 97) <= 0x19u )
          {
            LODWORD(v27) = (char)v27 - 97;
          }
          else if ( (unsigned __int8)(v27 - 65) <= 0x19u )
          {
            LODWORD(v27) = (char)v27 - 39;
          }
          else if ( (unsigned __int8)(v27 - 48) <= 9u )
          {
            LODWORD(v27) = (char)v27 + 4;
          }
          else
          {
            LODWORD(v27) = ((_BYTE)v27 != 46) + 62;
          }
          LODWORD(v26) = 6;
          goto LABEL_43;
        }
        v26 = *(_QWORD *)v53;
        if ( *(_QWORD *)v53 )
        {
LABEL_43:
          ++v24;
          sub_1524D80(v8, v27, v26);
          if ( v25 == v24 )
            goto LABEL_21;
        }
        else
        {
LABEL_35:
          if ( v25 == ++v24 )
            goto LABEL_21;
        }
      }
      if ( *(_QWORD *)v53 )
        sub_1525280(v8, v27, *(_QWORD *)v53);
      goto LABEL_35;
    }
    sub_1524E40(v8, a4 - v10, 6);
    if ( (_DWORD)a4 == v10 )
      goto LABEL_22;
    v48 = v12;
    v12 = (unsigned __int64)v8;
    v29 = v10;
    while ( 1 )
    {
      v31 = *(unsigned int *)(a3 + 4LL * v29);
      v32 = (*(_BYTE *)(v53 + 8) >> 1) & 7;
      if ( v32 == 2 )
        break;
      if ( v32 == 4 )
      {
        if ( (unsigned __int8)(v31 - 97) <= 0x19u )
        {
          LODWORD(v31) = (char)v31 - 97;
        }
        else if ( (unsigned __int8)(v31 - 65) <= 0x19u )
        {
          LODWORD(v31) = (char)v31 - 39;
        }
        else if ( (unsigned __int8)(v31 - 48) <= 9u )
        {
          LODWORD(v31) = (char)v31 + 4;
        }
        else
        {
          LODWORD(v31) = ((_BYTE)v31 != 46) + 62;
        }
        LODWORD(v30) = 6;
        goto LABEL_66;
      }
      v30 = *(_QWORD *)v53;
      if ( *(_QWORD *)v53 )
      {
LABEL_66:
        ++v29;
        sub_1524D80((_DWORD *)v12, v31, v30);
        if ( (_DWORD)a4 == v29 )
        {
LABEL_67:
          v10 = v29;
          v11 = 0;
          v8 = (_QWORD *)v12;
          LODWORD(v12) = v48;
          goto LABEL_22;
        }
      }
      else
      {
LABEL_58:
        if ( (_DWORD)a4 == ++v29 )
          goto LABEL_67;
      }
    }
    if ( *(_QWORD *)v53 )
      sub_1525280((_DWORD *)v12, v31, *(_QWORD *)v53);
    goto LABEL_58;
  }
}
