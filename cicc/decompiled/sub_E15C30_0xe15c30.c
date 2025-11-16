// Function: sub_E15C30
// Address: 0xe15c30
//
__int64 __fastcall sub_E15C30(__int64 a1, __int64 *a2)
{
  char *v4; // rdi
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  _BYTE *v11; // rdi
  unsigned __int64 v12; // rax
  void *v13; // rdi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax

  v4 = (char *)*a2;
  v5 = a2[1];
  if ( v4[v5 - 1] != 93 )
  {
    sub_E12F20(a2, 1u, " ");
    v5 = a2[1];
    v4 = (char *)*a2;
  }
  v6 = a2[2];
  if ( v5 + 1 > v6 )
  {
    v7 = v5 + 993;
    v8 = 2 * v6;
    if ( v7 <= v8 )
      a2[2] = v8;
    else
      a2[2] = v7;
    v9 = realloc(v4);
    *a2 = v9;
    v4 = (char *)v9;
    if ( !v9 )
      goto LABEL_18;
    v5 = a2[1];
  }
  v4[v5] = 91;
  v10 = a2[1] + 1;
  a2[1] = v10;
  v11 = *(_BYTE **)(a1 + 24);
  if ( v11 )
  {
    sub_E15BE0(v11, (__int64)a2);
    v10 = a2[1];
  }
  v12 = a2[2];
  v13 = (void *)*a2;
  if ( v10 + 1 > v12 )
  {
    v14 = v10 + 993;
    v15 = 2 * v12;
    if ( v14 > v15 )
      a2[2] = v14;
    else
      a2[2] = v15;
    v16 = realloc(v13);
    *a2 = v16;
    v13 = (void *)v16;
    if ( v16 )
    {
      v10 = a2[1];
      goto LABEL_15;
    }
LABEL_18:
    abort();
  }
LABEL_15:
  *((_BYTE *)v13 + v10) = 93;
  ++a2[1];
  return (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 16) + 40LL))(*(_QWORD *)(a1 + 16), a2);
}
