// Function: sub_E13910
// Address: 0xe13910
//
__int64 __fastcall sub_E13910(__int64 a1, __int64 *a2)
{
  _BYTE *v4; // rdi
  char v5; // al
  _BYTE *v6; // r12
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  char *v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  char *v13; // rdi
  char v15; // al

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 24) + 32LL))(*(_QWORD *)(a1 + 24));
  v4 = *(_BYTE **)(a1 + 24);
  v5 = v4[10];
  if ( (v5 & 3) == 2 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v4 + 8LL))(v4, a2) )
      goto LABEL_3;
    v4 = *(_BYTE **)(a1 + 24);
    v5 = v4[10];
  }
  else if ( (v5 & 3) == 0 )
  {
LABEL_3:
    sub_E12F20(a2, 1u, "(");
    goto LABEL_4;
  }
  v15 = v5 & 0xC;
  if ( v15 == 8 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v4 + 16LL))(v4, a2) )
      goto LABEL_3;
  }
  else if ( !v15 )
  {
    goto LABEL_3;
  }
  sub_E12F20(a2, 1u, " ");
LABEL_4:
  v6 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v6 + 32LL))(v6, a2);
  if ( (v6[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v6 + 40LL))(v6, a2);
  v7 = a2[1];
  v8 = a2[2];
  v9 = (char *)*a2;
  if ( v7 + 3 > v8 )
  {
    v10 = v7 + 995;
    v11 = 2 * v8;
    if ( v10 <= v11 )
      a2[2] = v11;
    else
      a2[2] = v10;
    v12 = realloc(v9);
    *a2 = v12;
    v9 = (char *)v12;
    if ( !v12 )
      abort();
    v7 = a2[1];
  }
  v13 = &v9[v7];
  *(_WORD *)v13 = 14906;
  v13[2] = 42;
  a2[1] += 3;
  return 14906;
}
