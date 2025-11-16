// Function: sub_E11740
// Address: 0xe11740
//
__int64 __fastcall sub_E11740(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // r12
  char *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax

  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  v6 = (__int64)*a2;
  if ( (unsigned __int64)(v4 + 2) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 994);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = (char *)v8;
    else
      a2[2] = (char *)v7;
    v9 = realloc((void *)v6);
    *a2 = (char *)v9;
    v6 = v9;
    if ( !v9 )
      goto LABEL_16;
    v4 = a2[1];
  }
  *(_WORD *)&v4[v6] = 15394;
  a2[1] += 2;
  v10 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 32LL))(v10, a2);
  if ( (v10[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v10 + 40LL))(v10, a2);
  v11 = a2[1];
  v12 = (unsigned __int64)a2[2];
  v13 = (__int64)*a2;
  if ( (unsigned __int64)(v11 + 2) > v12 )
  {
    v14 = (unsigned __int64)(v11 + 994);
    v15 = 2 * v12;
    if ( v14 > v15 )
      a2[2] = (char *)v14;
    else
      a2[2] = (char *)v15;
    v16 = realloc((void *)v13);
    *a2 = (char *)v16;
    v13 = v16;
    if ( v16 )
    {
      v11 = a2[1];
      goto LABEL_13;
    }
LABEL_16:
    abort();
  }
LABEL_13:
  *(_WORD *)&v11[v13] = 8766;
  a2[1] += 2;
  return 8766;
}
