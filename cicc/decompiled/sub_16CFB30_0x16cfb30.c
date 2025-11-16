// Function: sub_16CFB30
// Address: 0x16cfb30
//
void __fastcall sub_16CFB30(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  int v6; // r15d
  void *v7; // rdx
  __int64 (*v8)(void); // rax
  _BYTE *v9; // rdi
  size_t v10; // rbx
  char *v11; // rsi
  _BYTE *v12; // rax
  unsigned int v13; // eax
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rax
  size_t v17; // rdx
  _BYTE *v18; // rbx

  if ( !a2 )
    return;
  v6 = sub_16CE270(a1, a2);
  sub_16CFB30(a1, *(_QWORD *)(*a1 + 24LL * (unsigned int)(v6 - 1) + 16), a3);
  v7 = *(void **)(a3 + 24);
  if ( *(_QWORD *)(a3 + 16) - (_QWORD)v7 <= 0xDu )
  {
    a3 = sub_16E7EE0(a3, "Included from ", 14);
  }
  else
  {
    qmemcpy(v7, "Included from ", 14);
    *(_QWORD *)(a3 + 24) += 14LL;
  }
  v8 = *(__int64 (**)(void))(**(_QWORD **)(*a1 + 24LL * (unsigned int)(v6 - 1)) + 16LL);
  if ( (char *)v8 != (char *)sub_12BCB10 )
  {
    v16 = v8();
    v9 = *(_BYTE **)(a3 + 24);
    v11 = (char *)v16;
    v12 = *(_BYTE **)(a3 + 16);
    v10 = v17;
    if ( v12 - v9 < v17 )
      goto LABEL_7;
    if ( !v17 )
      goto LABEL_8;
LABEL_17:
    memcpy(v9, v11, v10);
    v18 = (_BYTE *)(*(_QWORD *)(a3 + 24) + v10);
    v12 = *(_BYTE **)(a3 + 16);
    *(_QWORD *)(a3 + 24) = v18;
    v9 = v18;
    goto LABEL_8;
  }
  v9 = *(_BYTE **)(a3 + 24);
  v10 = 14;
  v11 = "Unknown buffer";
  if ( *(_QWORD *)(a3 + 16) - (_QWORD)v9 > 0xDu )
    goto LABEL_17;
LABEL_7:
  a3 = sub_16E7EE0(a3, v11, v10);
  v12 = *(_BYTE **)(a3 + 16);
  v9 = *(_BYTE **)(a3 + 24);
LABEL_8:
  if ( v12 == v9 )
  {
    a3 = sub_16E7EE0(a3, ":", 1);
  }
  else
  {
    *v9 = 58;
    ++*(_QWORD *)(a3 + 24);
  }
  v13 = sub_16CFA40(a1, a2, v6);
  v14 = sub_16E7A90(a3, v13);
  v15 = *(_WORD **)(v14 + 24);
  if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
  {
    sub_16E7EE0(v14, ":\n", 2);
  }
  else
  {
    *v15 = 2618;
    *(_QWORD *)(v14 + 24) += 2LL;
  }
}
