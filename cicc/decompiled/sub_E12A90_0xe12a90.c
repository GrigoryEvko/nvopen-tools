// Function: sub_E12A90
// Address: 0xe12a90
//
unsigned __int64 __fastcall sub_E12A90(__int64 a1, char **a2)
{
  _BYTE *v4; // r12
  char *v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  _BYTE *v12; // r12
  unsigned __int64 result; // rax
  char *v14; // rdi
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax

  v4 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v4 + 32LL))(v4);
  if ( (v4[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v4 + 40LL))(v4, a2);
  v5 = a2[1];
  v6 = (unsigned __int64)a2[2];
  v7 = (__int64)*a2;
  if ( (unsigned __int64)(v5 + 8) > v6 )
  {
    v8 = (unsigned __int64)(v5 + 1000);
    v9 = 2 * v6;
    if ( v8 > v9 )
      a2[2] = (char *)v8;
    else
      a2[2] = (char *)v9;
    v10 = realloc((void *)v7);
    *a2 = (char *)v10;
    v7 = v10;
    if ( !v10 )
      goto LABEL_20;
    v5 = a2[1];
  }
  *(_QWORD *)&v5[v7] = 0x5B726F7463657620LL;
  v11 = (__int64)(a2[1] + 8);
  a2[1] = (char *)v11;
  v12 = *(_BYTE **)(a1 + 24);
  if ( v12 )
  {
    (*(void (__fastcall **)(_QWORD, char **))(*(_QWORD *)v12 + 32LL))(*(_QWORD *)(a1 + 24), a2);
    if ( (v12[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 40LL))(v12, a2);
    v11 = (__int64)a2[1];
  }
  result = (unsigned __int64)a2[2];
  v14 = *a2;
  if ( v11 + 1 > result )
  {
    v15 = v11 + 993;
    v16 = 2 * result;
    if ( v15 <= v16 )
      a2[2] = (char *)v16;
    else
      a2[2] = (char *)v15;
    result = realloc(v14);
    *a2 = (char *)result;
    v14 = (char *)result;
    if ( result )
    {
      v11 = (__int64)a2[1];
      goto LABEL_17;
    }
LABEL_20:
    abort();
  }
LABEL_17:
  v14[v11] = 93;
  ++a2[1];
  return result;
}
