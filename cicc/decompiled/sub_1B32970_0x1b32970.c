// Function: sub_1B32970
// Address: 0x1b32970
//
void __fastcall sub_1B32970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // rax
  const void *v8; // r9
  size_t v9; // r14
  int v10; // r15d
  _BYTE *v11; // rdi
  int v12; // eax
  unsigned int v13; // r14d
  int v14; // r15d
  _QWORD *v15; // rax
  _QWORD *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r10
  _QWORD *v19; // r13
  _QWORD *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  char v23; // si
  int v24; // r15d
  char v25; // dl
  __int64 v26; // r14
  __int64 *v27; // rax
  __int64 *v28; // rcx
  unsigned int v29; // edi
  __int64 *v30; // rsi
  _QWORD *v31; // rax
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // r13
  _QWORD *v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // r15
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdi
  _QWORD *v44; // rdx
  _QWORD *v45; // rax
  __int64 v47; // [rsp+20h] [rbp-250h]
  const void *v48; // [rsp+20h] [rbp-250h]
  _BYTE *v50; // [rsp+30h] [rbp-240h] BYREF
  __int64 v51; // [rsp+38h] [rbp-238h]
  _BYTE v52[560]; // [rsp+40h] [rbp-230h] BYREF

  v7 = *(unsigned int *)(a2 + 280);
  v8 = *(const void **)(a2 + 272);
  v50 = v52;
  v9 = 8 * v7;
  v10 = v7;
  v51 = 0x4000000000LL;
  if ( v7 > 0x40 )
  {
    v48 = v8;
    sub_16CD150((__int64)&v50, v52, v7, 8, a5, (int)v8);
    v8 = v48;
    v11 = &v50[8 * (unsigned int)v51];
  }
  else
  {
    v11 = v52;
    if ( !v9 )
      goto LABEL_3;
  }
  memcpy(v11, v8, v9);
  LODWORD(v9) = v51;
  v11 = v50;
LABEL_3:
  v12 = v10 + v9;
  v13 = 0;
  LODWORD(v51) = v12;
  v14 = v12;
  if ( v12 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD **)(a3 + 16);
      v16 = *(_QWORD **)(a3 + 8);
      v17 = v13;
      v18 = *(_QWORD *)&v11[8 * v13];
      if ( v15 == v16 )
      {
        v19 = &v16[*(unsigned int *)(a3 + 28)];
        if ( v16 == v19 )
        {
          v45 = *(_QWORD **)(a3 + 8);
        }
        else
        {
          do
          {
            if ( v18 == *v16 )
              break;
            ++v16;
          }
          while ( v19 != v16 );
          v45 = v19;
        }
        goto LABEL_70;
      }
      v47 = *(_QWORD *)&v11[8 * v13];
      v19 = &v15[*(unsigned int *)(a3 + 24)];
      v20 = sub_16CC9F0(a3, v18);
      v18 = v47;
      v17 = v13;
      v16 = v20;
      if ( v47 == *v20 )
        break;
      v21 = *(_QWORD *)(a3 + 16);
      if ( v21 == *(_QWORD *)(a3 + 8) )
      {
        v16 = (_QWORD *)(v21 + 8LL * *(unsigned int *)(a3 + 28));
        v45 = v16;
LABEL_70:
        while ( v45 != v16 && *v16 >= 0xFFFFFFFFFFFFFFFELL )
          ++v16;
        goto LABEL_8;
      }
      v16 = (_QWORD *)(v21 + 8LL * *(unsigned int *)(a3 + 24));
LABEL_8:
      if ( v16 == v19 )
      {
LABEL_17:
        ++v13;
        v11 = v50;
        if ( v13 == v14 )
          goto LABEL_18;
      }
      else
      {
        if ( v18 != a5 )
        {
          v22 = *(_QWORD *)(v18 + 48);
          if ( !v22 )
            goto LABEL_84;
          v23 = *(_BYTE *)(v22 - 8);
          if ( v23 != 55 )
          {
LABEL_15:
            if ( v23 == 54 && a1 == *(_QWORD *)(v22 - 48) )
              goto LABEL_17;
            goto LABEL_13;
          }
          while ( *(_QWORD *)(v22 - 48) != a1 )
          {
LABEL_13:
            v22 = *(_QWORD *)(v22 + 8);
            if ( !v22 )
LABEL_84:
              BUG();
            v23 = *(_BYTE *)(v22 - 8);
            if ( v23 != 55 )
              goto LABEL_15;
          }
        }
        --v14;
        *(_QWORD *)&v50[8 * v17] = *(_QWORD *)&v50[8 * (unsigned int)v51 - 8];
        v11 = v50;
        LODWORD(v51) = v51 - 1;
        if ( v13 == v14 )
        {
LABEL_18:
          v24 = v51;
          while ( 1 )
          {
            if ( !v24 )
              goto LABEL_77;
            v26 = *(_QWORD *)&v11[8 * v24 - 8];
            v27 = *(__int64 **)(a4 + 8);
            LODWORD(v51) = v24 - 1;
            if ( *(__int64 **)(a4 + 16) == v27 )
            {
              v28 = &v27[*(unsigned int *)(a4 + 28)];
              v29 = *(_DWORD *)(a4 + 28);
              if ( v27 != v28 )
              {
                v30 = 0;
                do
                {
                  while ( 1 )
                  {
                    if ( v26 == *v27 )
                      goto LABEL_20;
                    if ( *v27 != -2 )
                      break;
                    v30 = v27;
                    if ( v27 + 1 == v28 )
                      goto LABEL_29;
                    ++v27;
                  }
                  ++v27;
                }
                while ( v28 != v27 );
                if ( v30 )
                {
LABEL_29:
                  *v30 = v26;
                  --*(_DWORD *)(a4 + 32);
                  ++*(_QWORD *)a4;
                  goto LABEL_30;
                }
              }
              if ( v29 < *(_DWORD *)(a4 + 24) )
                break;
            }
            sub_16CCBA0(a4, v26);
            if ( v25 )
              goto LABEL_30;
LABEL_20:
            v24 = v51;
            v11 = v50;
          }
          *(_DWORD *)(a4 + 28) = v29 + 1;
          *v28 = v26;
          ++*(_QWORD *)a4;
          do
          {
LABEL_30:
            v26 = *(_QWORD *)(v26 + 8);
            if ( !v26 )
              goto LABEL_20;
            v31 = sub_1648700(v26);
          }
          while ( (unsigned __int8)(*((_BYTE *)v31 + 16) - 25) > 9u );
LABEL_43:
          v35 = v31[5];
          v36 = *(_QWORD **)(a3 + 16);
          v37 = *(_QWORD **)(a3 + 8);
          if ( v36 == v37 )
          {
            v38 = &v37[*(unsigned int *)(a3 + 28)];
            if ( v37 == v38 )
            {
              v44 = *(_QWORD **)(a3 + 8);
            }
            else
            {
              do
              {
                if ( v35 == *v37 )
                  break;
                ++v37;
              }
              while ( v38 != v37 );
              v44 = v38;
            }
          }
          else
          {
            v38 = &v36[*(unsigned int *)(a3 + 24)];
            v37 = sub_16CC9F0(a3, v35);
            if ( v35 == *v37 )
            {
              v40 = *(_QWORD *)(a3 + 16);
              if ( v40 == *(_QWORD *)(a3 + 8) )
                v41 = *(unsigned int *)(a3 + 28);
              else
                v41 = *(unsigned int *)(a3 + 24);
              v44 = (_QWORD *)(v40 + 8 * v41);
            }
            else
            {
              v39 = *(_QWORD *)(a3 + 16);
              if ( v39 != *(_QWORD *)(a3 + 8) )
              {
                v37 = (_QWORD *)(v39 + 8LL * *(unsigned int *)(a3 + 24));
LABEL_47:
                if ( v37 == v38 && a5 != v35 )
                {
                  v34 = (unsigned int)v51;
                  if ( (unsigned int)v51 >= HIDWORD(v51) )
                  {
                    sub_16CD150((__int64)&v50, v52, 0, 8, v32, v33);
                    v34 = (unsigned int)v51;
                  }
                  *(_QWORD *)&v50[8 * v34] = v35;
                  LODWORD(v51) = v51 + 1;
                }
                while ( 1 )
                {
                  v26 = *(_QWORD *)(v26 + 8);
                  if ( !v26 )
                    goto LABEL_20;
                  v31 = sub_1648700(v26);
                  if ( (unsigned __int8)(*((_BYTE *)v31 + 16) - 25) <= 9u )
                    goto LABEL_43;
                }
              }
              v37 = (_QWORD *)(v39 + 8LL * *(unsigned int *)(a3 + 28));
              v44 = v37;
            }
          }
          while ( v44 != v37 && *v37 >= 0xFFFFFFFFFFFFFFFELL )
            ++v37;
          goto LABEL_47;
        }
      }
    }
    v42 = *(_QWORD *)(a3 + 16);
    if ( v42 == *(_QWORD *)(a3 + 8) )
      v43 = *(unsigned int *)(a3 + 28);
    else
      v43 = *(unsigned int *)(a3 + 24);
    v45 = (_QWORD *)(v42 + 8 * v43);
    goto LABEL_70;
  }
LABEL_77:
  if ( v11 != v52 )
    _libc_free((unsigned __int64)v11);
}
