// Function: sub_391BFA0
// Address: 0x391bfa0
//
void __fastcall sub_391BFA0(__int64 a1, __int64 a2, unsigned __int64 a3, int a4, unsigned int a5)
{
  unsigned __int64 v8; // r13
  __int64 v9; // rcx
  char v10; // si
  char v11; // al
  char *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  size_t v15; // r13
  __int64 v16; // r14
  size_t v17; // r15
  char v18; // si
  char v19; // al
  char *v20; // rax
  __int64 v21; // r15
  unsigned __int64 v22; // rax
  char *v23; // rdi
  size_t v24; // r13
  size_t v25; // r14
  char v26; // si
  char v27; // cl
  __int64 v28; // r8
  unsigned __int64 v29; // rax
  char *v30; // rdi
  char v31; // si
  unsigned __int8 v32; // al
  __int64 v33; // rdx
  unsigned __int64 v34; // r13
  char v35; // si
  char v36; // al
  char *v37; // rax
  __int64 v38; // rdi
  _BYTE *v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // r13
  char v42; // si
  char v43; // al
  char *v44; // rax
  __int64 v45; // rdi
  char v46; // dl
  char *v47; // rax
  __int64 v48; // rdi
  _BYTE *v49; // rax
  __int64 v50; // rdx
  unsigned __int64 v51; // r13
  char v52; // si
  char v53; // al
  char *v54; // rax
  __int64 v55; // rdi
  char v56; // dl
  char *v57; // rax
  __int64 v58; // rdi
  char v59; // dl
  char *v60; // rax
  __int64 v61; // [rsp+10h] [rbp-70h]
  __int64 v62; // [rsp+18h] [rbp-68h]
  void *srcb; // [rsp+20h] [rbp-60h]
  char *src; // [rsp+20h] [rbp-60h]
  char *srca; // [rsp+20h] [rbp-60h]
  void *srcc; // [rsp+20h] [rbp-60h]
  void *srcd; // [rsp+20h] [rbp-60h]
  void *srce; // [rsp+20h] [rbp-60h]
  unsigned int v69; // [rsp+28h] [rbp-58h]
  _QWORD v71[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( a3 )
  {
    v8 = a3;
    v69 = (unsigned int)(a4 + 0xFFFF) >> 16;
    sub_391B370(a1, (__int64)v71, 2);
    v9 = *(_QWORD *)(a1 + 8);
    while ( 1 )
    {
      v10 = v8 & 0x7F;
      v11 = v8 & 0x7F | 0x80;
      v8 >>= 7;
      if ( v8 )
        v10 = v11;
      v12 = *(char **)(v9 + 24);
      if ( (unsigned __int64)v12 < *(_QWORD *)(v9 + 16) )
      {
        *(_QWORD *)(v9 + 24) = v12 + 1;
        *v12 = v10;
        if ( !v8 )
          goto LABEL_8;
      }
      else
      {
        srcb = (void *)v9;
        sub_16E7DE0(v9, v10);
        v9 = (__int64)srcb;
        if ( !v8 )
        {
LABEL_8:
          v13 = 7 * a3;
          v14 = a2;
          v62 = a2 + 8 * v13;
          if ( a2 == v62 )
            goto LABEL_40;
LABEL_9:
          v15 = *(_QWORD *)(v14 + 8);
          v16 = *(_QWORD *)(a1 + 8);
          src = *(char **)v14;
          v17 = v15;
          do
          {
            while ( 1 )
            {
              v18 = v17 & 0x7F;
              v19 = v17 & 0x7F | 0x80;
              v17 >>= 7;
              if ( v17 )
                v18 = v19;
              v20 = *(char **)(v16 + 24);
              if ( (unsigned __int64)v20 >= *(_QWORD *)(v16 + 16) )
                break;
              *(_QWORD *)(v16 + 24) = v20 + 1;
              *v20 = v18;
              if ( !v17 )
                goto LABEL_15;
            }
            sub_16E7DE0(v16, v18);
          }
          while ( v17 );
LABEL_15:
          v21 = *(_QWORD *)(a1 + 8);
          v22 = *(_QWORD *)(v21 + 16);
          v23 = *(char **)(v21 + 24);
          if ( v15 > v22 - (unsigned __int64)v23 )
          {
            sub_16E7EE0(*(_QWORD *)(a1 + 8), src, v15);
            v21 = *(_QWORD *)(a1 + 8);
            v23 = *(char **)(v21 + 24);
            v22 = *(_QWORD *)(v21 + 16);
          }
          else if ( v15 )
          {
            memcpy(v23, src, v15);
            *(_QWORD *)(v21 + 24) += v15;
            v21 = *(_QWORD *)(a1 + 8);
            v23 = *(char **)(v21 + 24);
            v22 = *(_QWORD *)(v21 + 16);
          }
          v24 = *(_QWORD *)(v14 + 24);
          srca = *(char **)(v14 + 16);
          v25 = v24;
          while ( 1 )
          {
            v26 = v25 & 0x7F;
            v27 = v25 & 0x7F | 0x80;
            v25 >>= 7;
            if ( v25 )
              v26 = v27;
            if ( (unsigned __int64)v23 < v22 )
            {
              *(_QWORD *)(v21 + 24) = v23 + 1;
              *v23 = v26;
              if ( !v25 )
                goto LABEL_25;
            }
            else
            {
              sub_16E7DE0(v21, v26);
              if ( !v25 )
              {
LABEL_25:
                v28 = *(_QWORD *)(a1 + 8);
                v29 = *(_QWORD *)(v28 + 16);
                v30 = *(char **)(v28 + 24);
                if ( v24 > v29 - (unsigned __int64)v30 )
                {
                  sub_16E7EE0(*(_QWORD *)(a1 + 8), srca, v24);
                  v28 = *(_QWORD *)(a1 + 8);
                  v30 = *(char **)(v28 + 24);
                  v29 = *(_QWORD *)(v28 + 16);
                }
                else if ( v24 )
                {
                  v61 = *(_QWORD *)(a1 + 8);
                  memcpy(v30, srca, v24);
                  *(_QWORD *)(v61 + 24) += v24;
                  v28 = *(_QWORD *)(a1 + 8);
                  v30 = *(char **)(v28 + 24);
                  v29 = *(_QWORD *)(v28 + 16);
                }
                v31 = *(_BYTE *)(v14 + 32);
                if ( v29 <= (unsigned __int64)v30 )
                {
                  sub_16E7DE0(v28, v31);
                }
                else
                {
                  *(_QWORD *)(v28 + 24) = v30 + 1;
                  *v30 = v31;
                }
                v32 = *(_BYTE *)(v14 + 32);
                if ( v32 == 2 )
                {
                  v38 = *(_QWORD *)(a1 + 8);
                  v39 = *(_BYTE **)(v38 + 24);
                  if ( (unsigned __int64)v39 >= *(_QWORD *)(v38 + 16) )
                  {
                    sub_16E7DE0(v38, 0);
                  }
                  else
                  {
                    *(_QWORD *)(v38 + 24) = v39 + 1;
                    *v39 = 0;
                  }
                  v40 = *(_QWORD *)(a1 + 8);
                  v41 = v69;
                  do
                  {
                    v42 = v41 & 0x7F;
                    v43 = v41 & 0x7F | 0x80;
                    v41 >>= 7;
                    if ( v41 )
                      v42 = v43;
                    v44 = *(char **)(v40 + 24);
                    if ( (unsigned __int64)v44 < *(_QWORD *)(v40 + 16) )
                    {
                      *(_QWORD *)(v40 + 24) = v44 + 1;
                      *v44 = v42;
                    }
                    else
                    {
                      srcd = (void *)v40;
                      sub_16E7DE0(v40, v42);
                      v40 = (__int64)srcd;
                    }
                  }
                  while ( v41 );
                  goto LABEL_39;
                }
                if ( v32 <= 2u )
                {
                  if ( v32 )
                  {
                    v45 = *(_QWORD *)(a1 + 8);
                    v46 = *(_BYTE *)(v14 + 36);
                    v47 = *(char **)(v45 + 24);
                    if ( (unsigned __int64)v47 >= *(_QWORD *)(v45 + 16) )
                    {
                      sub_16E7DE0(v45, v46);
                    }
                    else
                    {
                      *(_QWORD *)(v45 + 24) = v47 + 1;
                      *v47 = v46;
                    }
                    v48 = *(_QWORD *)(a1 + 8);
                    v49 = *(_BYTE **)(v48 + 24);
                    if ( (unsigned __int64)v49 >= *(_QWORD *)(v48 + 16) )
                    {
                      sub_16E7DE0(v48, 0);
                    }
                    else
                    {
                      *(_QWORD *)(v48 + 24) = v49 + 1;
                      *v49 = 0;
                    }
                    v50 = *(_QWORD *)(a1 + 8);
                    v51 = a5;
                    do
                    {
                      v52 = v51 & 0x7F;
                      v53 = v51 & 0x7F | 0x80;
                      v51 >>= 7;
                      if ( v51 )
                        v52 = v53;
                      v54 = *(char **)(v50 + 24);
                      if ( (unsigned __int64)v54 < *(_QWORD *)(v50 + 16) )
                      {
                        *(_QWORD *)(v50 + 24) = v54 + 1;
                        *v54 = v52;
                      }
                      else
                      {
                        srce = (void *)v50;
                        sub_16E7DE0(v50, v52);
                        v50 = (__int64)srce;
                      }
                    }
                    while ( v51 );
                  }
                  else
                  {
                    v33 = *(_QWORD *)(a1 + 8);
                    v34 = *(unsigned int *)(v14 + 36);
                    do
                    {
                      while ( 1 )
                      {
                        v35 = v34 & 0x7F;
                        v36 = v34 & 0x7F | 0x80;
                        v34 >>= 7;
                        if ( v34 )
                          v35 = v36;
                        v37 = *(char **)(v33 + 24);
                        if ( (unsigned __int64)v37 >= *(_QWORD *)(v33 + 16) )
                          break;
                        *(_QWORD *)(v33 + 24) = v37 + 1;
                        *v37 = v35;
                        if ( !v34 )
                          goto LABEL_39;
                      }
                      srcc = (void *)v33;
                      sub_16E7DE0(v33, v35);
                      v33 = (__int64)srcc;
                    }
                    while ( v34 );
                  }
LABEL_39:
                  v14 += 56;
                  if ( v62 == v14 )
                    goto LABEL_40;
                  goto LABEL_9;
                }
                v55 = *(_QWORD *)(a1 + 8);
                v56 = *(_BYTE *)(v14 + 36);
                v57 = *(char **)(v55 + 24);
                if ( (unsigned __int64)v57 >= *(_QWORD *)(v55 + 16) )
                {
                  sub_16E7DE0(v55, v56);
                }
                else
                {
                  *(_QWORD *)(v55 + 24) = v57 + 1;
                  *v57 = v56;
                }
                v58 = *(_QWORD *)(a1 + 8);
                v59 = *(_BYTE *)(v14 + 37);
                v60 = *(char **)(v58 + 24);
                if ( (unsigned __int64)v60 < *(_QWORD *)(v58 + 16) )
                {
                  v14 += 56;
                  *(_QWORD *)(v58 + 24) = v60 + 1;
                  *v60 = v59;
                  if ( v62 == v14 )
                    goto LABEL_40;
                  goto LABEL_9;
                }
                v14 += 56;
                sub_16E7DE0(v58, v59);
                if ( v62 == v14 )
                {
LABEL_40:
                  sub_3919EA0(a1, v71);
                  return;
                }
                goto LABEL_9;
              }
            }
            v23 = *(char **)(v21 + 24);
            v22 = *(_QWORD *)(v21 + 16);
          }
        }
      }
    }
  }
}
