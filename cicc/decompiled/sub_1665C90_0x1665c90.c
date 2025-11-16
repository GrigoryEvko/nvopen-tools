// Function: sub_1665C90
// Address: 0x1665c90
//
void __fastcall sub_1665C90(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  const char *v5; // rax
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rax
  const char *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v3 = *(_QWORD *)(a2 - 8);
    if ( *(_BYTE *)(**(_QWORD **)v3 + 8LL) == 15 )
      goto LABEL_3;
LABEL_8:
    v11 = 1;
    v5 = "Indirectbr operand must have pointer type!";
    goto LABEL_9;
  }
  v3 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( *(_BYTE *)(**(_QWORD **)v3 + 8LL) != 15 )
    goto LABEL_8;
LABEL_3:
  LODWORD(v4) = 0;
  do
  {
    if ( (_DWORD)v4 == (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) - 1 )
    {
      sub_1665790(a1, a2);
      return;
    }
    v4 = (unsigned int)(v4 + 1);
  }
  while ( *(_BYTE *)(**(_QWORD **)(v3 + 24 * v4) + 8LL) == 7 );
  v11 = 1;
  v5 = "Indirectbr destinations must all have pointer type!";
LABEL_9:
  v6 = *a1;
  v9 = v5;
  v10 = 3;
  if ( v6 )
  {
    sub_16E2CE0(&v9, v6);
    v7 = *(_BYTE **)(v6 + 24);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
    {
      sub_16E7DE0(v6, 10);
    }
    else
    {
      *(_QWORD *)(v6 + 24) = v7 + 1;
      *v7 = 10;
    }
    v8 = *a1;
    *((_BYTE *)a1 + 72) = 1;
    if ( v8 )
      sub_164FA80(a1, a2);
  }
  else
  {
    *((_BYTE *)a1 + 72) = 1;
  }
}
