// Function: sub_FFBC40
// Address: 0xffbc40
//
void __fastcall sub_FFBC40(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // r13
  char *v7; // rdi
  signed __int64 v8; // r12
  void *v9; // rax
  bool v10; // zf

  if ( !*(_BYTE *)(a1 + 560) )
    return;
  sub_FFBC00(a1, a2);
  if ( !*(_QWORD *)(a1 + 544) )
  {
    v3 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)(a1 + 552) == 0;
    *(_QWORD *)(a1 + 528) = v3;
    v4 = v3;
    if ( !v10 )
      goto LABEL_4;
LABEL_11:
    *(_QWORD *)(a1 + 536) = v4;
    v5 = v4;
    goto LABEL_5;
  }
  v3 = *(_QWORD *)(a1 + 528);
  v4 = *(unsigned int *)(a1 + 8);
  if ( !*(_QWORD *)(a1 + 552) )
    goto LABEL_11;
LABEL_4:
  v5 = *(_QWORD *)(a1 + 536);
LABEL_5:
  v6 = v5;
  v7 = *(char **)a1;
  if ( v3 <= v5 )
    v6 = v3;
  v8 = 32 * v4 - 32 * v6;
  if ( &v7[32 * v6] != &v7[32 * v4] )
  {
    v9 = memmove(v7, &v7[32 * v6], v8);
    v3 = *(_QWORD *)(a1 + 528);
    v5 = *(_QWORD *)(a1 + 536);
    v8 = (signed __int64)v9 + v8 - *(_QWORD *)a1;
  }
  *(_DWORD *)(a1 + 8) = v8 >> 5;
  *(_QWORD *)(a1 + 528) = v3 - v6;
  *(_QWORD *)(a1 + 536) = v5 - v6;
}
