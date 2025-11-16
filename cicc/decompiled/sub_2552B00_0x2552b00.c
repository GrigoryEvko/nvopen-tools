// Function: sub_2552B00
// Address: 0x2552b00
//
__int64 __fastcall sub_2552B00(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // r14
  __int64 v4; // rsi
  unsigned __int64 v5; // rdi
  __int64 **v7; // r12
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdi
  int v13; // r9d
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 *v17; // r15
  char v18; // al
  __int64 *v19; // rax
  __int64 v20; // r15
  __int64 v21; // rdx
  int v22; // eax
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 *v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+0h] [rbp-40h]
  __int64 *v31; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 - 88) = off_4A190D8;
  v2 = *(unsigned int *)(a1 + 168);
  *(_QWORD *)a1 = &unk_4A19168;
  sub_C7D6A0(*(_QWORD *)(a1 + 152), 16 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 8LL * *(unsigned int *)(a1 + 136), 8);
  *(_QWORD *)(a1 - 88) = &unk_4A18FE8;
  *(_QWORD *)a1 = &unk_4A19078;
  if ( !*(_DWORD *)(a1 + 104) )
  {
    v3 = *(_QWORD *)(a1 + 88);
    v4 = 0;
    goto LABEL_3;
  }
  if ( !byte_4FEF2C0[0] && (unsigned int)sub_2207590((__int64)byte_4FEF2C0) )
  {
    unk_4FEF2E0 = -4096;
    unk_4FEF2E8 = -4096;
    qword_4FEF2F0 = 0;
    unk_4FEF2F8 = 0;
    sub_2207640((__int64)byte_4FEF2C0);
  }
  if ( !byte_4FEF280[0] && (unsigned int)sub_2207590((__int64)byte_4FEF280) )
  {
    unk_4FEF2A0 = -8192;
    unk_4FEF2A8 = -8192;
    qword_4FEF2B0 = 0;
    unk_4FEF2B8 = 0;
    sub_2207640((__int64)byte_4FEF280);
  }
  v7 = *(__int64 ***)(a1 + 88);
  v4 = *(unsigned int *)(a1 + 104);
  v3 = (__int64)&v7[v4];
  if ( v7 != &v7[v4] )
  {
    while ( 1 )
    {
      v8 = *v7;
      v9 = **v7;
      v10 = (*v7)[1];
      if ( unk_4FEF2E8 != v10 || unk_4FEF2E0 != v9 )
        goto LABEL_11;
      v11 = v8[2];
      v12 = qword_4FEF2F0;
      if ( v11 != qword_4FEF2F0 )
      {
        if ( v11 == -4096 || qword_4FEF2F0 == -8192 || qword_4FEF2F0 == -4096 || v11 == -8192 )
          goto LABEL_11;
        if ( v11 )
        {
          v13 = *(_DWORD *)(v11 + 20) - *(_DWORD *)(v11 + 24);
          if ( qword_4FEF2F0 )
          {
            if ( v13 == *(_DWORD *)(qword_4FEF2F0 + 20) - *(_DWORD *)(qword_4FEF2F0 + 24) )
            {
              if ( v13 )
              {
                v14 = *(__int64 **)(v11 + 8);
                v15 = (__int64)(*(_BYTE *)(v11 + 28)
                              ? &v14[*(unsigned int *)(v11 + 20)]
                              : &v14[*(unsigned int *)(v11 + 16)]);
                if ( v14 != (__int64 *)v15 )
                {
                  while ( 1 )
                  {
                    v16 = *v14;
                    v17 = v14;
                    if ( (unsigned __int64)*v14 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( (__int64 *)v15 == ++v14 )
                      goto LABEL_13;
                  }
                  while ( (__int64 *)v15 != v17 )
                  {
                    v29 = (__int64 *)v15;
                    v18 = sub_B19060(v12, v16, v15, v9);
                    v15 = (__int64)v29;
                    if ( !v18 )
                    {
                      v8 = *v7;
                      v9 = **v7;
                      v10 = (*v7)[1];
                      goto LABEL_11;
                    }
                    v19 = v17 + 1;
                    if ( v17 + 1 == v29 )
                      goto LABEL_13;
                    while ( 1 )
                    {
                      v16 = *v19;
                      v17 = v19;
                      if ( (unsigned __int64)*v19 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( v29 == ++v19 )
                        goto LABEL_13;
                    }
                  }
                }
              }
              goto LABEL_13;
            }
LABEL_11:
            if ( unk_4FEF2A8 == v10 && unk_4FEF2A0 == v9 )
            {
              v20 = qword_4FEF2B0;
              v21 = v8[2];
              if ( v21 != qword_4FEF2B0
                && v21 != -4096
                && qword_4FEF2B0 != -8192
                && qword_4FEF2B0 != -4096
                && v21 != -8192 )
              {
                v22 = 0;
                if ( v21 )
                  v22 = *(_DWORD *)(v21 + 20) - *(_DWORD *)(v21 + 24);
                if ( qword_4FEF2B0 )
                {
                  v23 = *(_DWORD *)(qword_4FEF2B0 + 20) - *(_DWORD *)(qword_4FEF2B0 + 24);
                  if ( v23 == v22 && v23 && v23 >= *(_DWORD *)(v21 + 20) - *(_DWORD *)(v21 + 24) )
                  {
                    v24 = *(__int64 **)(v21 + 8);
                    v25 = (__int64)(*(_BYTE *)(v21 + 28)
                                  ? &v24[*(unsigned int *)(v21 + 20)]
                                  : &v24[*(unsigned int *)(v21 + 16)]);
                    if ( v24 != (__int64 *)v25 )
                    {
                      while ( 1 )
                      {
                        v26 = *v24;
                        v27 = (__int64)v24;
                        if ( (unsigned __int64)*v24 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( (__int64 *)v25 == ++v24 )
                          goto LABEL_13;
                      }
                      while ( v25 != v27 )
                      {
                        v30 = v27;
                        v31 = (__int64 *)v25;
                        if ( !(unsigned __int8)sub_B19060(v20, v26, v27, v25) )
                          break;
                        v25 = (__int64)v31;
                        v28 = (__int64 *)(v30 + 8);
                        if ( (__int64 *)(v30 + 8) == v31 )
                          break;
                        while ( 1 )
                        {
                          v26 = *v28;
                          v27 = (__int64)v28;
                          if ( (unsigned __int64)*v28 < 0xFFFFFFFFFFFFFFFELL )
                            break;
                          if ( v31 == ++v28 )
                            goto LABEL_13;
                        }
                      }
                    }
                  }
                }
              }
            }
            goto LABEL_13;
          }
          if ( v13 )
            goto LABEL_11;
        }
        else if ( qword_4FEF2F0 && *(_DWORD *)(qword_4FEF2F0 + 24) != *(_DWORD *)(qword_4FEF2F0 + 20) )
        {
          goto LABEL_11;
        }
      }
LABEL_13:
      if ( (__int64 **)v3 == ++v7 )
      {
        v3 = *(_QWORD *)(a1 + 88);
        v4 = *(unsigned int *)(a1 + 104);
        break;
      }
    }
  }
LABEL_3:
  sub_C7D6A0(v3, v4 * 8, 8);
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 != a1 + 32 )
    _libc_free(v5);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  return sub_254FD20(a1 - 80);
}
