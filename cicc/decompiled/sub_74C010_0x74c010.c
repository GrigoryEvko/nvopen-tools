// Function: sub_74C010
// Address: 0x74c010
//
void __fastcall sub_74C010(__int64 a1, char a2, __int64 a3)
{
  char v5; // al
  const char *v6; // rdi
  void (__fastcall *v7)(const char *, __int64); // rdx
  void (__fastcall *v8)(const char *, __int64); // rax
  __int64 v9; // rdi
  unsigned int v10; // r14d
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  char v13; // al
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // r14
  char v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rax
  char v20; // r14
  char v21; // [rsp+0h] [rbp-70h] BYREF
  char v22; // [rsp+1h] [rbp-6Fh]

  v5 = *(_BYTE *)(a1 + 89);
  if ( (v5 & 0x40) != 0 || ((v5 & 8) != 0 ? (v6 = *(const char **)(a1 + 24)) : (v6 = *(const char **)(a1 + 8)), !v6) )
  {
    v8 = *(void (__fastcall **)(const char *, __int64))a3;
    if ( a2 == 6
      && (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u
      && (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 109LL) & 0x20) != 0 )
    {
      v10 = dword_4F07588;
      v8("lambda []", a3);
      if ( v10 )
      {
        v19 = sub_72F130((_QWORD *)a1);
        if ( v19 )
        {
          v20 = *(_BYTE *)(a3 + 155);
          *(_BYTE *)(a3 + 155) = 1;
          sub_74B930(*(_QWORD *)(v19 + 152), a3);
          *(_BYTE *)(a3 + 155) = v20;
        }
      }
      else
      {
        (*(void (__fastcall **)(const char *, __int64))a3)(" type at line ", a3);
        v11 = *(unsigned int *)(a1 + 64);
        if ( (unsigned int)v11 > 9 )
        {
          sub_622470(v11, &v21);
        }
        else
        {
          v22 = 0;
          v21 = v11 + 48;
        }
        (*(void (__fastcall **)(char *, __int64))a3)(&v21, a3);
        (*(void (__fastcall **)(const char *, __int64))a3)(", col. ", a3);
        v12 = *(unsigned __int16 *)(a1 + 68);
        if ( (unsigned __int16)v12 > 9u )
        {
          sub_622470(v12, &v21);
        }
        else
        {
          v22 = 0;
          v21 = v12 + 48;
        }
        (*(void (__fastcall **)(char *, __int64))a3)(&v21, a3);
      }
    }
    else
    {
      v8("<unnamed", a3);
      (*(void (__fastcall **)(char *, __int64))a3)(">", a3);
    }
  }
  else
  {
    v7 = *(void (__fastcall **)(const char *, __int64))a3;
    if ( a2 != 11 )
      goto LABEL_6;
    if ( *(_BYTE *)(a1 + 174) != 3 )
    {
      if ( (*(_BYTE *)(a1 + 194) & 0x40) != 0 )
      {
        v14 = a1;
        do
          v14 = *(_QWORD *)(v14 + 232);
        while ( (*(_BYTE *)(v14 + 194) & 0x40) != 0 );
        v6 = 0;
        if ( (*(_BYTE *)(v14 + 89) & 0x40) == 0 )
        {
          if ( (*(_BYTE *)(v14 + 89) & 8) != 0 )
            v6 = *(const char **)(v14 + 24);
          else
            v6 = *(const char **)(v14 + 8);
        }
      }
LABEL_6:
      v7(v6, a3);
      if ( dword_4F072C8 )
        return;
      goto LABEL_11;
    }
    v16 = *(_QWORD *)(a1 + 152);
    v17 = *(_BYTE *)(a3 + 149);
    v7("operator ", a3);
    for ( ; *(_BYTE *)(v16 + 140) == 12; v16 = *(_QWORD *)(v16 + 160) )
      ;
    *(_BYTE *)(a3 + 149) = 1;
    v18 = *(_QWORD *)(v16 + 160);
    *(_BYTE *)(a3 + 149) = v17;
    sub_74B930(v18, a3);
  }
  if ( dword_4F072C8 )
    return;
LABEL_11:
  if ( *(_BYTE *)(a3 + 152) )
    return;
  if ( a2 == 6 )
  {
    v13 = *(_BYTE *)(a1 + 140);
    if ( (unsigned __int8)(v13 - 9) <= 2u )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
    }
    else
    {
      if ( v13 != 12 )
        return;
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8LL);
    }
  }
  else
  {
    if ( a2 == 7 )
    {
      v15 = *(__int64 **)(a1 + 216);
      if ( v15 )
      {
        v9 = *v15;
        if ( *v15 )
          goto LABEL_30;
      }
      return;
    }
    if ( !dword_4D0460C || a2 != 11 )
      return;
    v9 = *(_QWORD *)(a1 + 240);
  }
  if ( v9 )
LABEL_30:
    sub_7477E0(v9, 0, a3);
}
