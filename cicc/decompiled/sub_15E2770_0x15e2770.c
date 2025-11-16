// Function: sub_15E2770
// Address: 0x15e2770
//
__int64 __fastcall sub_15E2770(char *s, void *a2, size_t a3)
{
  unsigned int v3; // eax
  size_t v4; // r13
  unsigned int *v5; // r12
  unsigned int v6; // r12d
  const char *v8; // r14
  size_t v9; // rax
  __int64 v10; // rbx
  const char *v11; // r13
  size_t v12; // rax
  size_t v13; // rdx
  const char *v14; // r13
  size_t v15; // rax
  const char *v16; // r13
  size_t v17; // rax
  const char *v18; // r13
  size_t v19; // rax
  size_t v20; // rax
  const char *v21; // r13
  size_t v22; // rax
  const char *v23; // r13
  size_t v24; // rax
  unsigned int *v25; // r13
  const char *v26; // r12
  size_t v27; // rax
  const char *v28; // r13
  size_t v29; // rax
  const char *v30; // r13
  size_t v31; // rax
  void *s2; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  char v34; // [rsp+17h] [rbp-39h] BYREF
  char *v35; // [rsp+18h] [rbp-38h] BYREF

  s2 = a2;
  n = a3;
  v35 = &v34;
  *(_QWORD *)(__readfsqword(0) - 24) = &v35;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_15DE590;
  if ( !&_pthread_key_create )
  {
    v3 = -1;
LABEL_87:
    sub_4264C5(v3);
  }
  v3 = pthread_once(&dword_4F9E14C, init_routine);
  if ( v3 )
    goto LABEL_87;
  if ( s )
  {
    v4 = strlen(s);
    v5 = (unsigned int *)sub_15DE660((__int64)&unk_429EF88 - 360, (__int64)&unk_429EF88, (__int64)&s2);
    if ( v5 == (unsigned int *)&unk_429EF88 )
      goto LABEL_5;
  }
  else
  {
    v5 = (unsigned int *)sub_15DE660((__int64)&unk_429EF88 - 360, (__int64)&unk_429EF88, (__int64)&s2);
    if ( v5 == (unsigned int *)&unk_429EF88 )
      return 0;
    v4 = 0;
  }
  v8 = &byte_4C7E1E0[v5[1]];
  v9 = strlen(v8);
  if ( n == v9 && (!n || !memcmp(v8, s2, n)) )
    return *v5;
LABEL_5:
  switch ( v4 )
  {
    case 7uLL:
      if ( *(_DWORD *)s == 1668440417 && *((_WORD *)s + 2) == 13928 && s[6] == 52 )
      {
        v5 = (unsigned int *)sub_15DE660((__int64)&unk_429EE18 - 24, (__int64)&unk_429EE18, (__int64)&s2);
        if ( v5 != (unsigned int *)&unk_429EE18 )
        {
          v21 = &byte_4C7E1E0[v5[1]];
          v22 = strlen(v21);
          if ( v22 == n && (!v22 || !memcmp(v21, s2, v22)) )
            return *v5;
        }
      }
      if ( *(_DWORD *)s != 1635280232 || *((_WORD *)s + 2) != 28519 || s[6] != 110 )
        return 0;
      v6 = 0;
      v10 = sub_15DE660((__int64)&unk_429E8E0 - 13440, (__int64)&unk_429E8E0, (__int64)&s2);
      if ( (_UNKNOWN *)v10 != &unk_429E8E0 )
      {
        v11 = &byte_4C7E1E0[*(unsigned int *)(v10 + 4)];
        v12 = strlen(v11);
        v13 = n;
        if ( n == v12 )
        {
          if ( n )
          {
LABEL_47:
            if ( memcmp(v11, s2, v13) )
              return v6;
            return *(unsigned int *)v10;
          }
          return *(unsigned int *)v10;
        }
      }
      break;
    case 6uLL:
      if ( *(_DWORD *)s == 1734634849 && *((_WORD *)s + 2) == 28259 )
      {
        v6 = 0;
        v10 = sub_15DE660((__int64)&unk_429EDF8 - 536, (__int64)&unk_429EDF8, (__int64)&s2);
        if ( (_UNKNOWN *)v10 == &unk_429EDF8 )
          return v6;
LABEL_44:
        v11 = &byte_4C7E1E0[*(unsigned int *)(v10 + 4)];
        v20 = strlen(v11);
        if ( v20 != n )
          return v6;
        if ( !v20 )
          return *(unsigned int *)v10;
        v13 = v20;
        goto LABEL_47;
      }
      return 0;
    case 3uLL:
      if ( *(_WORD *)s != 29281
        || s[2] != 109
        || (v5 = (unsigned int *)sub_15DE660((__int64)&unk_429EBE0 - 736, (__int64)&unk_429EBE0, (__int64)&s2),
            v5 == (unsigned int *)&unk_429EBE0)
        || (v23 = &byte_4C7E1E0[v5[1]], v24 = strlen(v23), v24 != n)
        || v24 && memcmp(v23, s2, v24) )
      {
        if ( *(_WORD *)s == 28770 && s[2] == 102 )
        {
          v25 = (unsigned int *)sub_15DE660((__int64)&unk_429E900 - 32, (__int64)&unk_429E900, (__int64)&s2);
          if ( v25 != (unsigned int *)&unk_429E900 )
          {
            v26 = &byte_4C7E1E0[v25[1]];
            v27 = strlen(v26);
            if ( n == v27 && (!n || !memcmp(v26, s2, n)) )
              return *v25;
          }
        }
        if ( *(_WORD *)s != 28784
          || s[2] != 99
          || (v5 = (unsigned int *)sub_15DE660((__int64)&unk_42989E0 - 2912, (__int64)&unk_42989E0, (__int64)&s2),
              v5 == (unsigned int *)&unk_42989E0)
          || (v28 = &byte_4C7E1E0[v5[1]], v29 = strlen(v28), v29 != n)
          || v29 && memcmp(v28, s2, v29) )
        {
          if ( *(_WORD *)s == 14456 && s[2] == 54 )
          {
            v6 = 0;
            v10 = sub_15DE660((__int64)&unk_4297928 - 9544, (__int64)&unk_4297928, (__int64)&s2);
            if ( (_UNKNOWN *)v10 == &unk_4297928 )
              return v6;
            goto LABEL_44;
          }
          return 0;
        }
      }
      return *v5;
    case 4uLL:
      if ( *(_DWORD *)s == 1936746861 )
      {
        v5 = (unsigned int *)sub_15DE660((__int64)&unk_429B458 - 5336, (__int64)&unk_429B458, (__int64)&s2);
        if ( v5 != (unsigned int *)&unk_429B458 )
        {
          v14 = &byte_4C7E1E0[v5[1]];
          v15 = strlen(v14);
          if ( v15 == n && (!v15 || !memcmp(v14, s2, v15)) )
            return *v5;
        }
      }
      if ( *(_DWORD *)s == 1836480110 )
      {
        v5 = (unsigned int *)sub_15DE660((__int64)&unk_4299F68 - 5512, (__int64)&unk_4299F68, (__int64)&s2);
        if ( v5 != (unsigned int *)&unk_4299F68 )
        {
          v16 = &byte_4C7E1E0[v5[1]];
          v17 = strlen(v16);
          if ( v17 == n && (!v17 || !memcmp(v16, s2, v17)) )
            return *v5;
        }
      }
      if ( *(_DWORD *)s == 808466034 )
      {
        v5 = (unsigned int *)sub_15DE660((__int64)&unk_4297E80 - 96, (__int64)&unk_4297E80, (__int64)&s2);
        if ( v5 != (unsigned int *)&unk_4297E80 )
        {
          v18 = &byte_4C7E1E0[v5[1]];
          v19 = strlen(v18);
          if ( v19 == n && (!v19 || !memcmp(v18, s2, v19)) )
            return *v5;
        }
      }
      if ( *(_DWORD *)s == 809055091 )
      {
        v6 = 0;
        v10 = sub_15DE660((__int64)&unk_4297E18 - 1240, (__int64)&unk_4297E18, (__int64)&s2);
        if ( (_UNKNOWN *)v10 == &unk_4297E18 )
          return v6;
        goto LABEL_44;
      }
      return 0;
    default:
      if ( v4 != 5 || *(_DWORD *)s != 1919902584 || s[4] != 101 )
        return 0;
      v6 = 0;
      v10 = sub_15DE660((__int64)&unk_42953E0 - 32, (__int64)&unk_42953E0, (__int64)&s2);
      if ( (_UNKNOWN *)v10 != &unk_42953E0 )
      {
        v30 = &byte_4C7E1E0[*(unsigned int *)(v10 + 4)];
        v31 = strlen(v30);
        if ( v31 == n )
        {
          if ( !v31 )
            return *(unsigned int *)v10;
          v6 = 0;
          if ( !memcmp(v30, s2, v31) )
            return *(unsigned int *)v10;
        }
      }
      break;
  }
  return v6;
}
