// Function: sub_E312A0
// Address: 0xe312a0
//
__int64 __fastcall sub_E312A0(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  void *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax

  v6 = a2[1];
  v7 = a2[2];
  v8 = (void *)*a2;
  if ( v6 + 1 > v7 )
  {
    v9 = v6 + 993;
    v10 = 2 * v7;
    if ( v9 <= v10 )
      a2[2] = v10;
    else
      a2[2] = v9;
    v11 = realloc(v8);
    *a2 = v11;
    v8 = (void *)v11;
    if ( !v11 )
      goto LABEL_15;
    v6 = a2[1];
  }
  *((_BYTE *)v8 + v6) = 91;
  ++a2[1];
  sub_E311C0(a1, a2, a3);
  v12 = a2[1];
  v13 = a2[2];
  if ( v12 + 1 <= v13 )
  {
    v16 = *a2;
    goto LABEL_12;
  }
  v14 = v12 + 993;
  v15 = 2 * v13;
  if ( v14 > v15 )
    a2[2] = v14;
  else
    a2[2] = v15;
  v16 = realloc((void *)*a2);
  *a2 = v16;
  if ( !v16 )
LABEL_15:
    abort();
  v12 = a2[1];
LABEL_12:
  *(_BYTE *)(v16 + v12) = 93;
  ++a2[1];
  return (*(__int64 (__fastcall **)(_QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 24) + 32LL))(
           *(_QWORD *)(a1 + 24),
           a2,
           a3);
}
