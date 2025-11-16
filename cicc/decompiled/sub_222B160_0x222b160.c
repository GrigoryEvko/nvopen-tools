// Function: sub_222B160
// Address: 0x222b160
//
__int64 __fastcall sub_222B160(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v5; // rax
  unsigned __int64 v6; // rdx
  __int64 result; // rax
  bool v8; // zf
  __int64 v9; // rbp
  __int64 v10; // rdi
  unsigned __int64 v11; // r12
  int v12; // eax
  signed __int64 v13; // r13
  signed __int64 v14; // rax
  const void *v15; // rsi
  signed __int64 v16; // rbp
  size_t v17; // r14
  signed __int64 v18; // rax
  char *v19; // r15
  __int64 v20; // rax
  char *v21; // rsi
  char *v22; // r9
  int v23; // r13d
  char v24; // r14
  __int64 v25; // rbp
  char v26; // al
  ssize_t v27; // rax
  __int64 v28; // rax
  size_t v29; // r8
  char *v30; // rsi
  size_t v31; // r13
  ssize_t v32; // rax
  int *v33; // rax
  unsigned __int8 *v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdi
  char *v37; // [rsp+0h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 120) & 8) == 0 )
    return 0xFFFFFFFFLL;
  if ( *(_BYTE *)(a1 + 170) )
  {
    a2 = 0xFFFFFFFFLL;
    result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 104LL))(a1, 0xFFFFFFFFLL);
    if ( (_DWORD)result == -1 )
      return result;
    v5 = *(unsigned __int8 **)(a1 + 152);
    v8 = *(_BYTE *)(a1 + 192) == 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 16) = v5;
    *(_QWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 48) = 0;
    *(_BYTE *)(a1 + 170) = 0;
    if ( v8 )
      goto LABEL_4;
LABEL_9:
    v8 = *(_QWORD *)(a1 + 8) == (_QWORD)v5;
    v6 = *(_QWORD *)(a1 + 184);
    *(_BYTE *)(a1 + 192) = 0;
    a4 = *(_QWORD *)(a1 + 152);
    v5 = (unsigned __int8 *)(*(_QWORD *)(a1 + 176) + !v8);
    *(_QWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 176) = v5;
    *(_QWORD *)(a1 + 8) = a4;
    *(_QWORD *)(a1 + 16) = v5;
    if ( (unsigned __int64)v5 < v6 )
      return *v5;
    goto LABEL_10;
  }
  v5 = *(unsigned __int8 **)(a1 + 16);
  if ( *(_BYTE *)(a1 + 192) )
    goto LABEL_9;
LABEL_4:
  v6 = *(_QWORD *)(a1 + 24);
  if ( (unsigned __int64)v5 < v6 )
    return *v5;
LABEL_10:
  v9 = 2;
  v10 = *(_QWORD *)(a1 + 200);
  if ( *(_QWORD *)(a1 + 160) >= 2u )
    v9 = *(_QWORD *)(a1 + 160);
  v11 = v9 - 1;
  if ( !v10 )
    sub_426219(0, a2, v6, a4);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v10 + 48LL))(v10) )
  {
    v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 40LL))(*(_QWORD *)(a1 + 200));
    if ( v12 <= 0 )
    {
      v13 = v9 + (*(int (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 64LL))(*(_QWORD *)(a1 + 200)) - 2;
      v14 = v9 - 1;
    }
    else
    {
      v13 = v11 * v12;
      v14 = v13;
    }
    v15 = *(const void **)(a1 + 224);
    v16 = *(_QWORD *)(a1 + 232) - (_QWORD)v15;
    v17 = v14 - v16;
    if ( v14 <= v16 )
      v17 = 0;
    v18 = *(_QWORD *)(a1 + 216);
    if ( *(_BYTE *)(a1 + 169) && *(_QWORD *)(a1 + 8) == *(_QWORD *)(a1 + 24) )
    {
      if ( !v16 )
      {
        if ( v13 <= v18 )
          goto LABEL_22;
        v19 = (char *)sub_2207820(v13);
        goto LABEL_62;
      }
      v17 = 0;
      if ( v13 <= v18 )
        goto LABEL_66;
      v17 = 0;
      v19 = (char *)sub_2207820(v13);
    }
    else
    {
      if ( v13 <= v18 )
      {
        if ( !v16 )
        {
LABEL_22:
          v19 = *(char **)(a1 + 208);
          goto LABEL_23;
        }
LABEL_66:
        memmove(*(void **)(a1 + 208), v15, *(_QWORD *)(a1 + 232) - (_QWORD)v15);
        goto LABEL_22;
      }
      v19 = (char *)sub_2207820(v13);
      if ( !v16 )
      {
LABEL_62:
        v36 = *(_QWORD *)(a1 + 208);
        if ( v36 )
          j_j___libc_free_0_0(v36);
        *(_QWORD *)(a1 + 208) = v19;
        *(_QWORD *)(a1 + 216) = v13;
LABEL_23:
        v20 = *(_QWORD *)(a1 + 132);
        v21 = &v19[v16];
        *(_QWORD *)(a1 + 224) = v19;
        *(_QWORD *)(a1 + 232) = &v19[v16];
        *(_QWORD *)(a1 + 140) = v20;
        if ( v17 )
        {
          v23 = 0;
          goto LABEL_31;
        }
        v22 = *(char **)(a1 + 8);
        v23 = 0;
        v37 = v22;
        if ( v21 <= v19 )
        {
          while ( 1 )
          {
LABEL_30:
            v21 = *(char **)(a1 + 232);
            v17 = 1;
            v16 = (signed __int64)&v21[-*(_QWORD *)(a1 + 208)];
LABEL_31:
            if ( (signed __int64)(v17 + v16) > *(_QWORD *)(a1 + 216) )
              sub_426A1E((__int64)"basic_filebuf::underflow codecvt::max_length() is not valid");
            v27 = sub_2207DA0((FILE **)(a1 + 104), v21, v17);
            if ( v27 )
            {
              if ( v27 == -1 )
                goto LABEL_50;
              v24 = 0;
            }
            else
            {
              v24 = 1;
            }
            v22 = *(char **)(a1 + 8);
            v19 = *(char **)(a1 + 224);
            v21 = (char *)(*(_QWORD *)(a1 + 232) + v27);
            *(_QWORD *)(a1 + 232) = v21;
            v37 = v22;
            if ( v21 > v19 )
              break;
            if ( v23 == 3 )
              goto LABEL_42;
            if ( v24 )
              goto LABEL_38;
          }
        }
        else
        {
          v24 = 0;
        }
        v23 = (*(__int64 (__fastcall **)(_QWORD, __int64, char *, char *, __int64, char *, char *, char **))(**(_QWORD **)(a1 + 200) + 32LL))(
                *(_QWORD *)(a1 + 200),
                a1 + 132,
                v19,
                v21,
                a1 + 224,
                v22,
                &v22[v11],
                &v37);
        if ( v23 != 3 )
        {
          v25 = (__int64)&v37[-*(_QWORD *)(a1 + 8)];
          if ( v23 != 2 )
          {
            v26 = v24 | (v37 != *(char **)(a1 + 8));
            goto LABEL_29;
          }
          if ( (__int64)&v37[-*(_QWORD *)(a1 + 8)] <= 0 )
          {
            if ( !v24 )
              sub_426A1E((__int64)"basic_filebuf::underflow invalid byte sequence in file");
            goto LABEL_58;
          }
          goto LABEL_52;
        }
        v21 = *(char **)(a1 + 232);
        v22 = *(char **)(a1 + 8);
LABEL_42:
        v29 = *(_QWORD *)(a1 + 208);
        v30 = &v21[-v29];
        v25 = (__int64)v30;
        v31 = (size_t)v30;
        if ( v11 < (unsigned __int64)v30 )
        {
          v25 = v11;
          v31 = v11;
        }
        else if ( !v30 )
        {
LABEL_44:
          *(_QWORD *)(a1 + 224) = v29;
          v23 = 3;
          v26 = v24 | (v25 != 0);
LABEL_29:
          if ( !v26 )
            goto LABEL_30;
          if ( v25 > 0 )
            goto LABEL_52;
          if ( v24 )
          {
LABEL_38:
            v28 = *(_QWORD *)(a1 + 152);
            *(_QWORD *)(a1 + 40) = 0;
            *(_QWORD *)(a1 + 32) = 0;
            *(_QWORD *)(a1 + 8) = v28;
            *(_QWORD *)(a1 + 16) = v28;
            *(_QWORD *)(a1 + 24) = v28;
            *(_QWORD *)(a1 + 48) = 0;
            *(_BYTE *)(a1 + 169) = 0;
            if ( v23 == 1 )
              sub_426A1E((__int64)"basic_filebuf::underflow incomplete character in file");
            return 0xFFFFFFFFLL;
          }
LABEL_50:
          v33 = __errno_location();
          sub_426AAD((__int64)"basic_filebuf::underflow error reading the file", *v33);
        }
        memcpy(v22, *(const void **)(a1 + 208), v31);
        v29 = v31 + *(_QWORD *)(a1 + 208);
        goto LABEL_44;
      }
    }
    memcpy(v19, *(const void **)(a1 + 224), v16);
    goto LABEL_62;
  }
  v32 = sub_2207DA0((FILE **)(a1 + 104), *(void **)(a1 + 8), v9 - 1);
  v25 = v32;
  if ( !v32 )
  {
LABEL_58:
    v35 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 8) = v35;
    *(_QWORD *)(a1 + 16) = v35;
    *(_QWORD *)(a1 + 24) = v35;
    *(_QWORD *)(a1 + 48) = 0;
    *(_BYTE *)(a1 + 169) = 0;
    return 0xFFFFFFFFLL;
  }
  if ( v32 <= 0 )
    goto LABEL_50;
LABEL_52:
  v34 = *(unsigned __int8 **)(a1 + 152);
  v8 = (*(_BYTE *)(a1 + 120) & 8) == 0;
  *(_QWORD *)(a1 + 8) = v34;
  *(_QWORD *)(a1 + 16) = v34;
  if ( v8 )
    *(_QWORD *)(a1 + 24) = v34;
  else
    *(_QWORD *)(a1 + 24) = &v34[v25];
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 169) = 1;
  return *v34;
}
