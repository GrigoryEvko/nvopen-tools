// Function: sub_E17890
// Address: 0xe17890
//
__int64 __fastcall sub_E17890(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rax
  void *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // rsi
  int v17; // r13d

  sub_E12F20((__int64 *)a2, 7u, "'lambda");
  sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 64), *(const void **)(a1 + 72));
  sub_E12F20((__int64 *)a2, 1u, "'");
  if ( *(_QWORD *)(a1 + 24) )
  {
    v17 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a2 + 32) = 0;
    sub_E12F20((__int64 *)a2, 1u, "<");
    sub_E161C0((_QWORD *)(a1 + 16), (char **)a2);
    sub_E12F20((__int64 *)a2, 1u, ">");
    *(_DWORD *)(a2 + 32) = v17;
  }
  if ( *(_QWORD *)(a1 + 32) )
  {
    sub_E12F20((__int64 *)a2, 0xAu, " requires ");
    sub_E15BE0(*(_BYTE **)(a1 + 32), a2);
    sub_E12F20((__int64 *)a2, 1u, " ");
  }
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_QWORD *)(a2 + 16);
  ++*(_DWORD *)(a2 + 32);
  v6 = *(void **)a2;
  v7 = v4 + 1;
  if ( v4 + 1 > v5 )
  {
    v8 = v4 + 993;
    v9 = 2 * v5;
    if ( v8 > v9 )
      *(_QWORD *)(a2 + 16) = v8;
    else
      *(_QWORD *)(a2 + 16) = v9;
    v10 = realloc(v6);
    *(_QWORD *)a2 = v10;
    v6 = (void *)v10;
    if ( !v10 )
      goto LABEL_20;
    v4 = *(_QWORD *)(a2 + 8);
    v7 = v4 + 1;
  }
  *(_QWORD *)(a2 + 8) = v7;
  *((_BYTE *)v6 + v4) = 40;
  sub_E161C0((_QWORD *)(a1 + 40), (char **)a2);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  --*(_DWORD *)(a2 + 32);
  if ( v11 + 1 > v12 )
  {
    v13 = v11 + 993;
    v14 = 2 * v12;
    if ( v13 > v14 )
      *(_QWORD *)(a2 + 16) = v13;
    else
      *(_QWORD *)(a2 + 16) = v14;
    result = realloc(*(void **)a2);
    *(_QWORD *)a2 = result;
    if ( result )
    {
      v16 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a2 + 8) = v16 + 1;
      *(_BYTE *)(result + v16) = 41;
      if ( !*(_QWORD *)(a1 + 56) )
        return result;
LABEL_17:
      sub_E12F20((__int64 *)a2, 0xAu, " requires ");
      return sub_E15BE0(*(_BYTE **)(a1 + 56), a2);
    }
LABEL_20:
    abort();
  }
  result = *(_QWORD *)a2;
  *(_QWORD *)(a2 + 8) = v11 + 1;
  *(_BYTE *)(result + v11) = 41;
  if ( *(_QWORD *)(a1 + 56) )
    goto LABEL_17;
  return result;
}
