// Function: sub_2EE68A0
// Address: 0x2ee68a0
//
char __fastcall sub_2EE68A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r14
  char v5; // r14
  int *v7; // rax
  int v8; // eax
  int v9; // ebx
  __int64 *v10; // rax
  char v11; // dl
  __int64 *v12; // rax
  char v13; // dl
  _DWORD *v14; // rax
  int v15; // ebx
  __int64 *v16; // rax
  char v17; // dl

  v4 = **(_QWORD **)(a1 + 32);
  if ( (unsigned __int8)sub_B2D610(v4, 47) )
    return 1;
  v5 = sub_B2D610(v4, 18);
  if ( v5 )
    return 1;
  if ( a3 == 0 || a2 == 0 )
    return v5;
  v7 = *(int **)(a2 + 8);
  if ( !v7 )
    return v5;
  if ( LOBYTE(qword_4F91668[8]) )
    return 1;
  v5 = qword_4F91BA8[8];
  if ( !LOBYTE(qword_4F91BA8[8]) )
    return v5;
  if ( LOBYTE(qword_4F919E8[8]) )
    goto LABEL_16;
  v8 = *v7;
  if ( !v8 )
  {
    if ( LOBYTE(qword_4F91908[8]) )
      goto LABEL_16;
    goto LABEL_12;
  }
  if ( v8 != 2 )
  {
LABEL_12:
    if ( !LOBYTE(qword_4F91AC8[8]) )
    {
LABEL_13:
      v9 = qword_4F91588[8];
      v10 = sub_2E39F50(a3, a1);
      if ( v11 )
        return !sub_D85370(a2, v9, (unsigned __int64)v10);
      return v5;
    }
LABEL_15:
    if ( !(unsigned __int8)sub_D84430(a2) )
      goto LABEL_16;
    goto LABEL_24;
  }
  if ( !(unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91828[8])
    || (unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91748[8]) )
  {
LABEL_16:
    v12 = sub_2E39F50(a3, a1);
    v5 = v13;
    if ( v13 )
      return sub_D84450(a2, (unsigned __int64)v12);
    return v5;
  }
  if ( LOBYTE(qword_4F91AC8[8]) )
    goto LABEL_15;
LABEL_24:
  v14 = *(_DWORD **)(a2 + 8);
  if ( !v14 || *v14 != 2 )
    goto LABEL_13;
  v15 = qword_4F914A8[8];
  v16 = sub_2E39F50(a3, a1);
  v5 = v17;
  if ( !v17 )
    return v5;
  return sub_D853A0(a2, v15, (unsigned __int64)v16);
}
