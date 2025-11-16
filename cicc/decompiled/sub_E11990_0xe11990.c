// Function: sub_E11990
// Address: 0xe11990
//
__int64 __fastcall sub_E11990(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rdi
  char v5; // al
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  void *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  char v13; // al

  v4 = *(_BYTE **)(a1 + 24);
  v5 = v4[10];
  if ( (v5 & 3) == 2 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 8LL))(v4) )
      goto LABEL_3;
    v4 = *(_BYTE **)(a1 + 24);
    v5 = v4[10];
  }
  else if ( (v5 & 3) == 0 )
  {
    goto LABEL_3;
  }
  v13 = v5 & 0xC;
  if ( v13 != 8 )
  {
    if ( v13 )
      return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
    goto LABEL_3;
  }
  if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 16LL))(v4, a2) )
  {
LABEL_3:
    v6 = *(_QWORD *)(a2 + 8);
    v7 = *(_QWORD *)(a2 + 16);
    v8 = *(void **)a2;
    if ( v6 + 1 > v7 )
    {
      v9 = v6 + 993;
      v10 = 2 * v7;
      if ( v9 > v10 )
        *(_QWORD *)(a2 + 16) = v9;
      else
        *(_QWORD *)(a2 + 16) = v10;
      v11 = realloc(v8);
      *(_QWORD *)a2 = v11;
      v8 = (void *)v11;
      if ( !v11 )
        abort();
      v6 = *(_QWORD *)(a2 + 8);
    }
    *((_BYTE *)v8 + v6) = 41;
    ++*(_QWORD *)(a2 + 8);
  }
  v4 = *(_BYTE **)(a1 + 24);
  return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
}
