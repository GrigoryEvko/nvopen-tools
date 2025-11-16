// Function: sub_2550870
// Address: 0x2550870
//
__int64 __fastcall sub_2550870(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi
  __int64 **v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // r15
  char v17; // al
  __int64 *v18; // rax
  __int64 v19; // r15
  __int64 v20; // rdx
  int v21; // eax
  unsigned int v22; // ecx
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 *v27; // rax
  __int64 *v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 *v30; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 - 88) = &unk_4A1D4C0;
  *(_QWORD *)a1 = &unk_4A1D550;
  if ( !*(_DWORD *)(a1 + 104) )
  {
    v2 = *(_QWORD *)(a1 + 88);
    v3 = 0;
    goto LABEL_3;
  }
  if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
  {
    unk_4FEF260 = -4096;
    unk_4FEF268 = -4096;
    unk_4FEF270 = 0;
    unk_4FEF278 = 0;
    sub_2207640((__int64)byte_4FEF240);
  }
  if ( !byte_4FEF208[0] && (unsigned int)sub_2207590((__int64)byte_4FEF208) )
  {
    unk_4FEF220 = -8192;
    unk_4FEF228 = -8192;
    qword_4FEF230 = 0;
    unk_4FEF238 = 0;
    sub_2207640((__int64)byte_4FEF208);
  }
  v6 = *(__int64 ***)(a1 + 88);
  v3 = *(unsigned int *)(a1 + 104);
  v2 = (__int64)&v6[v3];
  if ( v6 != &v6[v3] )
  {
    while ( 1 )
    {
      v7 = *v6;
      v8 = **v6;
      v9 = (*v6)[1];
      if ( unk_4FEF268 != v9 || unk_4FEF260 != v8 )
        goto LABEL_11;
      v10 = v7[2];
      v11 = unk_4FEF270;
      if ( v10 != unk_4FEF270 )
      {
        if ( v10 == -4096 || unk_4FEF270 == -8192 || unk_4FEF270 == -4096 || v10 == -8192 )
          goto LABEL_11;
        if ( v10 )
        {
          v12 = *(_DWORD *)(v10 + 20) - *(_DWORD *)(v10 + 24);
          if ( unk_4FEF270 )
          {
            if ( v12 == *(_DWORD *)(unk_4FEF270 + 20LL) - *(_DWORD *)(unk_4FEF270 + 24LL) )
            {
              if ( v12 )
              {
                v13 = *(__int64 **)(v10 + 8);
                v14 = (__int64)(*(_BYTE *)(v10 + 28)
                              ? &v13[*(unsigned int *)(v10 + 20)]
                              : &v13[*(unsigned int *)(v10 + 16)]);
                if ( v13 != (__int64 *)v14 )
                {
                  while ( 1 )
                  {
                    v15 = *v13;
                    v16 = v13;
                    if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( (__int64 *)v14 == ++v13 )
                      goto LABEL_13;
                  }
                  while ( (__int64 *)v14 != v16 )
                  {
                    v28 = (__int64 *)v14;
                    v17 = sub_B19060(v11, v15, v14, v8);
                    v14 = (__int64)v28;
                    if ( !v17 )
                    {
                      v7 = *v6;
                      v8 = **v6;
                      v9 = (*v6)[1];
                      goto LABEL_11;
                    }
                    v18 = v16 + 1;
                    if ( v16 + 1 == v28 )
                      goto LABEL_13;
                    while ( 1 )
                    {
                      v15 = *v18;
                      v16 = v18;
                      if ( (unsigned __int64)*v18 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v28 == ++v18 )
                        goto LABEL_13;
                    }
                  }
                }
              }
              goto LABEL_13;
            }
LABEL_11:
            if ( unk_4FEF220 == v8 && unk_4FEF228 == v9 )
            {
              v19 = qword_4FEF230;
              v20 = v7[2];
              if ( v20 != qword_4FEF230
                && v20 != -4096
                && qword_4FEF230 != -8192
                && qword_4FEF230 != -4096
                && v20 != -8192 )
              {
                v21 = 0;
                if ( v20 )
                  v21 = *(_DWORD *)(v20 + 20) - *(_DWORD *)(v20 + 24);
                if ( qword_4FEF230 )
                {
                  v22 = *(_DWORD *)(qword_4FEF230 + 20) - *(_DWORD *)(qword_4FEF230 + 24);
                  if ( v22 )
                  {
                    if ( v22 == v21 && v22 >= *(_DWORD *)(v20 + 20) - *(_DWORD *)(v20 + 24) )
                    {
                      v23 = *(__int64 **)(v20 + 8);
                      v24 = (__int64)(*(_BYTE *)(v20 + 28)
                                    ? &v23[*(unsigned int *)(v20 + 20)]
                                    : &v23[*(unsigned int *)(v20 + 16)]);
                      if ( v23 != (__int64 *)v24 )
                      {
                        while ( 1 )
                        {
                          v25 = *v23;
                          v26 = (__int64)v23;
                          if ( (unsigned __int64)*v23 < 0xFFFFFFFFFFFFFFFELL )
                            break;
                          if ( (__int64 *)v24 == ++v23 )
                            goto LABEL_13;
                        }
                        while ( v24 != v26 )
                        {
                          v29 = v26;
                          v30 = (__int64 *)v24;
                          if ( !(unsigned __int8)sub_B19060(v19, v25, v26, v24) )
                            break;
                          v24 = (__int64)v30;
                          v27 = (__int64 *)(v29 + 8);
                          if ( (__int64 *)(v29 + 8) == v30 )
                            break;
                          while ( 1 )
                          {
                            v25 = *v27;
                            v26 = (__int64)v27;
                            if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
                              break;
                            if ( v30 == ++v27 )
                              goto LABEL_13;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            goto LABEL_13;
          }
          if ( v12 )
            goto LABEL_11;
        }
        else if ( unk_4FEF270 && *(_DWORD *)(unk_4FEF270 + 20LL) != *(_DWORD *)(unk_4FEF270 + 24LL) )
        {
          goto LABEL_11;
        }
      }
LABEL_13:
      if ( (__int64 **)v2 == ++v6 )
      {
        v2 = *(_QWORD *)(a1 + 88);
        v3 = *(unsigned int *)(a1 + 104);
        break;
      }
    }
  }
LABEL_3:
  sub_C7D6A0(v2, v3 * 8, 8);
  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 != a1 + 32 )
    _libc_free(v4);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  return sub_254FD20(a1 - 80);
}
