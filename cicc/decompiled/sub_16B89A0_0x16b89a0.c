// Function: sub_16B89A0
// Address: 0x16b89a0
//
__int64 sub_16B89A0()
{
  __int64 v0; // rax
  __int64 v1; // r14
  __int64 v2; // r13
  __int64 *v3; // rax
  __int64 v4; // rax
  int v5; // ecx
  __int64 v6; // r15
  __int64 *v7; // rax
  __int64 v8; // rax
  int v9; // ecx
  _QWORD *v10; // rsi
  size_t **v11; // rax
  size_t *v12; // rdx
  size_t **v13; // r12
  size_t **v14; // rbx
  __int64 v15; // rsi
  size_t *v16; // rdx
  size_t **v17; // rax
  _QWORD *v19; // rsi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r15
  __int64 *v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rsi
  unsigned int v29; // edi
  __int64 *v30; // rcx
  __int64 *v31; // rsi
  unsigned int v32; // edi
  __int64 *v33; // rcx
  __int64 v34; // [rsp+8h] [rbp-38h]

  v0 = sub_22077B0(320);
  v1 = v0;
  if ( !v0 )
    return v1;
  *(_QWORD *)(v0 + 8) = 0;
  *(_QWORD *)v0 = v0 + 16;
  *(_QWORD *)(v0 + 80) = v0 + 112;
  *(_QWORD *)(v0 + 88) = v0 + 112;
  v34 = v0 + 240;
  *(_BYTE *)(v0 + 16) = 0;
  *(_QWORD *)(v0 + 32) = 0;
  *(_QWORD *)(v0 + 40) = 0;
  *(_QWORD *)(v0 + 48) = 0;
  *(_QWORD *)(v0 + 56) = 0;
  *(_QWORD *)(v0 + 64) = 0;
  *(_QWORD *)(v0 + 72) = 0;
  *(_QWORD *)(v0 + 96) = 16;
  *(_DWORD *)(v0 + 104) = 0;
  *(_QWORD *)(v0 + 240) = 0;
  *(_QWORD *)(v0 + 248) = v0 + 280;
  *(_QWORD *)(v0 + 256) = v0 + 280;
  *(_QWORD *)(v0 + 264) = 4;
  *(_DWORD *)(v0 + 272) = 0;
  *(_QWORD *)(v0 + 312) = 0;
  v2 = sub_16B4B80((__int64)&unk_4FA0190);
  v3 = *(__int64 **)(v1 + 248);
  if ( *(__int64 **)(v1 + 256) != v3 )
  {
LABEL_3:
    sub_16CCBA0(v34, v2);
    goto LABEL_4;
  }
  v28 = &v3[*(unsigned int *)(v1 + 268)];
  v29 = *(_DWORD *)(v1 + 268);
  if ( v3 == v28 )
  {
LABEL_66:
    if ( v29 < *(_DWORD *)(v1 + 264) )
    {
      *(_DWORD *)(v1 + 268) = v29 + 1;
      *v28 = v2;
      ++*(_QWORD *)(v1 + 240);
      goto LABEL_4;
    }
    goto LABEL_3;
  }
  v30 = 0;
  while ( v2 != *v3 )
  {
    if ( *v3 == -2 )
      v30 = v3;
    if ( v28 == ++v3 )
    {
      if ( !v30 )
        goto LABEL_66;
      *v30 = v2;
      --*(_DWORD *)(v1 + 272);
      ++*(_QWORD *)(v1 + 240);
      break;
    }
  }
LABEL_4:
  if ( v2 != sub_16B4B80((__int64)&unk_4FA0170) )
  {
    v4 = sub_16B4B80((__int64)&unk_4FA0170);
    v5 = *(_DWORD *)(v4 + 136);
    if ( v5 )
    {
      v19 = *(_QWORD **)(v4 + 128);
      if ( *v19 != -8 && *v19 )
      {
        v22 = *(__int64 **)(v4 + 128);
      }
      else
      {
        v20 = v19 + 1;
        do
        {
          do
          {
            v21 = *v20;
            v22 = v20++;
          }
          while ( v21 == -8 );
        }
        while ( !v21 );
      }
      v23 = &v19[v5];
      while ( v23 != v22 )
      {
        v24 = *(_QWORD *)(*v22 + 8);
        if ( ((*(_WORD *)(v24 + 12) >> 7) & 3) == 1
          || (*(_BYTE *)(v24 + 13) & 8) != 0
          || (*(_BYTE *)(v24 + 12) & 7) == 4
          || *(_QWORD *)(v24 + 32) )
        {
          sub_16B85B0((const char **)v1, v24, v2);
        }
        else
        {
          sub_16B7D30(v1, v24, v2, (const char *)(*v22 + 16), *(_QWORD *)*v22);
        }
        v25 = v22[1];
        if ( v25 != -8 && v25 )
        {
          ++v22;
        }
        else
        {
          v26 = v22 + 2;
          do
          {
            do
            {
              v27 = *v26;
              v22 = v26++;
            }
            while ( !v27 );
          }
          while ( v27 == -8 );
        }
      }
    }
  }
  v6 = sub_16B4B80((__int64)&unk_4FA0170);
  v7 = *(__int64 **)(v1 + 248);
  if ( *(__int64 **)(v1 + 256) != v7 )
    goto LABEL_7;
  v31 = &v7[*(unsigned int *)(v1 + 268)];
  v32 = *(_DWORD *)(v1 + 268);
  if ( v7 == v31 )
  {
LABEL_64:
    if ( v32 >= *(_DWORD *)(v1 + 264) )
    {
LABEL_7:
      sub_16CCBA0(v34, v6);
      goto LABEL_8;
    }
    *(_DWORD *)(v1 + 268) = v32 + 1;
    *v31 = v6;
    ++*(_QWORD *)(v1 + 240);
  }
  else
  {
    v33 = 0;
    while ( v6 != *v7 )
    {
      if ( *v7 == -2 )
        v33 = v7;
      if ( v31 == ++v7 )
      {
        if ( !v33 )
          goto LABEL_64;
        *v33 = v6;
        --*(_DWORD *)(v1 + 272);
        ++*(_QWORD *)(v1 + 240);
        break;
      }
    }
  }
LABEL_8:
  if ( v6 != sub_16B4B80((__int64)&unk_4FA0170) )
  {
    v8 = sub_16B4B80((__int64)&unk_4FA0170);
    v9 = *(_DWORD *)(v8 + 136);
    if ( v9 )
    {
      v10 = *(_QWORD **)(v8 + 128);
      if ( *v10 != -8 && *v10 )
      {
        v13 = *(size_t ***)(v8 + 128);
      }
      else
      {
        v11 = (size_t **)(v10 + 1);
        do
        {
          do
          {
            v12 = *v11;
            v13 = v11++;
          }
          while ( !v12 );
        }
        while ( v12 == (size_t *)-8LL );
      }
      v14 = (size_t **)&v10[v9];
      if ( v14 != v13 )
      {
        while ( 1 )
        {
          v15 = (*v13)[1];
          if ( ((*(_WORD *)(v15 + 12) >> 7) & 3) == 1
            || (*(_BYTE *)(v15 + 13) & 8) != 0
            || (*(_BYTE *)(v15 + 12) & 7) == 4
            || *(_QWORD *)(v15 + 32) )
          {
            sub_16B85B0((const char **)v1, v15, v6);
          }
          else
          {
            sub_16B7D30(v1, v15, v6, (const char *)*v13 + 16, **v13);
          }
          v16 = v13[1];
          v17 = v13 + 1;
          if ( v16 == (size_t *)-8LL )
            break;
LABEL_23:
          if ( !v16 )
            goto LABEL_22;
          if ( v14 == v17 )
            return v1;
          v13 = v17;
        }
        do
        {
LABEL_22:
          v16 = v17[1];
          ++v17;
        }
        while ( v16 == (size_t *)-8LL );
        goto LABEL_23;
      }
    }
  }
  return v1;
}
