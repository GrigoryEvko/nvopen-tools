// Function: sub_E161C0
// Address: 0xe161c0
//
void __fastcall sub_E161C0(_QWORD *a1, char **a2)
{
  __int64 v4; // rbx
  __int64 i; // r12
  _BYTE *v6; // r14
  __int64 v7; // r14
  unsigned __int64 v8; // rax
  char *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  void *v14; // rdi
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  char *v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  char *v21; // rcx
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  char v25; // [rsp-41h] [rbp-41h]
  __int64 v26; // [rsp-40h] [rbp-40h]

  if ( a1[1] )
  {
    v4 = 0;
    v25 = 1;
    v26 = (__int64)a2[1];
    for ( i = v26; ; a2[1] = (char *)i )
    {
      while ( 1 )
      {
        v6 = *(_BYTE **)(*a1 + 8 * v4);
        if ( (unsigned int)((char)(4 * v6[9]) >> 2) <= 0x11 )
          break;
        v12 = (unsigned __int64)a2[2];
        v13 = i + 1;
        ++*((_DWORD *)a2 + 8);
        v14 = *a2;
        if ( i + 1 <= v12 )
        {
          v17 = i;
        }
        else
        {
          v15 = 2 * v12;
          if ( i + 993 > v15 )
            a2[2] = (char *)(i + 993);
          else
            a2[2] = (char *)v15;
          v16 = realloc(v14);
          *a2 = (char *)v16;
          v14 = (void *)v16;
          if ( !v16 )
LABEL_35:
            abort();
          v17 = (__int64)a2[1];
          v13 = v17 + 1;
        }
        a2[1] = (char *)v13;
        *((_BYTE *)v14 + v17) = 40;
        (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v6 + 32LL))(v6, a2);
        if ( (v6[9] & 0xC0) != 0x40 )
          (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v6 + 40LL))(v6, a2);
        v18 = a2[1];
        v19 = (unsigned __int64)a2[2];
        --*((_DWORD *)a2 + 8);
        v20 = (__int64)*a2;
        v21 = v18 + 1;
        if ( (unsigned __int64)(v18 + 1) > v19 )
        {
          v22 = (unsigned __int64)(v18 + 993);
          v23 = 2 * v19;
          if ( v22 > v23 )
            a2[2] = (char *)v22;
          else
            a2[2] = (char *)v23;
          v24 = realloc((void *)v20);
          *a2 = (char *)v24;
          v20 = v24;
          if ( !v24 )
            goto LABEL_35;
          v18 = a2[1];
          v21 = v18 + 1;
        }
        a2[1] = v21;
        ++v4;
        v18[v20] = 41;
        v7 = (__int64)a2[1];
        if ( v7 != i )
          goto LABEL_7;
LABEL_26:
        a2[1] = (char *)v26;
        if ( a1[1] == v4 )
          return;
        i = v26;
        if ( !v25 )
        {
          v7 = v26;
          goto LABEL_8;
        }
      }
      (*(void (__fastcall **)(_QWORD, char **))(*(_QWORD *)v6 + 32LL))(*(_QWORD *)(*a1 + 8 * v4), a2);
      if ( (v6[9] & 0xC0) != 0x40 )
        (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v6 + 40LL))(v6, a2);
      v7 = (__int64)a2[1];
      ++v4;
      if ( v7 == i )
        goto LABEL_26;
LABEL_7:
      if ( a1[1] == v4 )
        return;
LABEL_8:
      v8 = (unsigned __int64)a2[2];
      v9 = *a2;
      if ( v7 + 2 > v8 )
      {
        v10 = 2 * v8;
        a2[2] = (char *)(v7 + 994 > v10 ? v7 + 994 : v10);
        v11 = realloc(v9);
        *a2 = (char *)v11;
        v9 = (char *)v11;
        if ( !v11 )
          goto LABEL_35;
      }
      v26 = v7;
      *(_WORD *)&v9[(_QWORD)a2[1]] = 8236;
      v25 = 0;
      i = (__int64)(a2[1] + 2);
    }
  }
}
