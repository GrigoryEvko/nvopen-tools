// Function: sub_E30E80
// Address: 0xe30e80
//
__int64 __fastcall sub_E30E80(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // rax

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(char **)a2;
  if ( v6 + 8 > v7 )
  {
    v9 = v6 + 1000;
    v10 = 2 * v7;
    if ( v9 <= v10 )
      *(_QWORD *)(a2 + 16) = v10;
    else
      *(_QWORD *)(a2 + 16) = v9;
    v11 = realloc(v8);
    *(_QWORD *)a2 = v11;
    v8 = (char *)v11;
    if ( !v11 )
      goto LABEL_15;
    v6 = *(_QWORD *)(a2 + 8);
  }
  *(_QWORD *)&v8[v6] = 0x726F74617265706FLL;
  *(_QWORD *)(a2 + 8) += 8LL;
  sub_E2EB40(a1, a2, a3);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(_QWORD *)(a2 + 16);
  if ( v12 + 1 <= v13 )
  {
    v16 = *(_QWORD *)a2;
    goto LABEL_12;
  }
  v14 = v12 + 993;
  v15 = 2 * v13;
  if ( v14 > v15 )
    *(_QWORD *)(a2 + 16) = v14;
  else
    *(_QWORD *)(a2 + 16) = v15;
  v16 = realloc(*(void **)a2);
  *(_QWORD *)a2 = v16;
  if ( !v16 )
LABEL_15:
    abort();
  v12 = *(_QWORD *)(a2 + 8);
LABEL_12:
  *(_BYTE *)(v16 + v12) = 32;
  ++*(_QWORD *)(a2 + 8);
  return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 24) + 16LL))(
           *(_QWORD *)(a1 + 24),
           a2,
           a3);
}
