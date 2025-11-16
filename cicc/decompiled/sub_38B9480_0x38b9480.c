// Function: sub_38B9480
// Address: 0x38b9480
//
void __fastcall sub_38B9480(__int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // r13
  char *v6; // rdi
  signed __int64 v7; // r12
  void *v8; // rax
  bool v9; // zf

  if ( !*(_BYTE *)(a1 + 304) )
    return;
  sub_38B9450(a1);
  v2 = *(unsigned int *)(a1 + 8);
  if ( !*(_QWORD *)(a1 + 288) )
  {
    v9 = *(_QWORD *)(a1 + 296) == 0;
    *(_QWORD *)(a1 + 272) = v2;
    v3 = v2;
    if ( !v9 )
      goto LABEL_4;
LABEL_11:
    *(_QWORD *)(a1 + 280) = v2;
    v4 = v2;
    goto LABEL_5;
  }
  v3 = *(_QWORD *)(a1 + 272);
  if ( !*(_QWORD *)(a1 + 296) )
    goto LABEL_11;
LABEL_4:
  v4 = *(_QWORD *)(a1 + 280);
LABEL_5:
  v5 = v4;
  v6 = *(char **)a1;
  if ( v3 <= v4 )
    v5 = v3;
  v7 = 16 * v2 - 16 * v5;
  if ( &v6[16 * v5] != &v6[16 * v2] )
  {
    v8 = memmove(v6, &v6[16 * v5], v7);
    v3 = *(_QWORD *)(a1 + 272);
    v4 = *(_QWORD *)(a1 + 280);
    v7 = (signed __int64)v8 + v7 - *(_QWORD *)a1;
  }
  *(_DWORD *)(a1 + 8) = v7 >> 4;
  *(_QWORD *)(a1 + 272) = v3 - v5;
  *(_QWORD *)(a1 + 280) = v4 - v5;
}
