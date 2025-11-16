// Function: sub_E17470
// Address: 0xe17470
//
unsigned __int64 __fastcall sub_E17470(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r12
  unsigned __int64 result; // rax
  char *v13; // rdi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  char *v16; // rsi
  int v17; // r13d

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(char **)a2;
  if ( v4 + 2 > v5 )
  {
    v7 = v4 + 994;
    v8 = 2 * v5;
    if ( v7 <= v8 )
      *(_QWORD *)(a2 + 16) = v8;
    else
      *(_QWORD *)(a2 + 16) = v7;
    v9 = realloc(v6);
    *(_QWORD *)a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_22;
    v4 = *(_QWORD *)(a2 + 8);
  }
  *(_WORD *)&v6[v4] = 23899;
  v10 = *(_QWORD *)(a2 + 8) + 2LL;
  *(_QWORD *)(a2 + 8) = v10;
  v11 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v11 + 8) == 52 )
  {
    if ( *(_QWORD *)(v11 + 24) )
    {
      v17 = *(_DWORD *)(a2 + 32);
      *(_DWORD *)(a2 + 32) = 0;
      sub_E12F20((__int64 *)a2, 1u, "<");
      sub_E161C0((_QWORD *)(v11 + 16), (char **)a2);
      sub_E12F20((__int64 *)a2, 1u, ">");
      *(_DWORD *)(a2 + 32) = v17;
    }
    if ( *(_QWORD *)(v11 + 32) )
    {
      sub_E12F20((__int64 *)a2, 0xAu, " requires ");
      sub_E15BE0(*(_BYTE **)(v11 + 32), a2);
      sub_E12F20((__int64 *)a2, 1u, " ");
    }
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    sub_E161C0((_QWORD *)(v11 + 40), (char **)a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
    if ( *(_QWORD *)(v11 + 56) )
    {
      sub_E12F20((__int64 *)a2, 0xAu, " requires ");
      sub_E15BE0(*(_BYTE **)(v11 + 56), a2);
    }
    v10 = *(_QWORD *)(a2 + 8);
  }
  result = *(_QWORD *)(a2 + 16);
  v13 = *(char **)a2;
  if ( v10 + 5 > result )
  {
    v14 = v10 + 997;
    v15 = 2 * result;
    if ( v14 <= v15 )
      *(_QWORD *)(a2 + 16) = v15;
    else
      *(_QWORD *)(a2 + 16) = v14;
    result = realloc(v13);
    *(_QWORD *)a2 = result;
    v13 = (char *)result;
    if ( result )
    {
      v10 = *(_QWORD *)(a2 + 8);
      goto LABEL_12;
    }
LABEL_22:
    abort();
  }
LABEL_12:
  v16 = &v13[v10];
  *(_DWORD *)v16 = 774778491;
  v16[4] = 125;
  *(_QWORD *)(a2 + 8) += 5LL;
  return result;
}
