// Function: sub_2EE6AD0
// Address: 0x2ee6ad0
//
char __fastcall sub_2EE6AD0(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 v5; // r13
  char v6; // r13
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 v10; // rsi
  int *v11; // rax
  int v12; // eax
  int v13; // r15d
  __int64 *v14; // rax
  char v15; // dl
  __int64 *v16; // rax
  char v17; // dl
  _DWORD *v18; // rax
  int v19; // r15d
  __int64 *v20; // rax
  char v21; // dl

  v5 = **(_QWORD **)(a1 + 32);
  if ( (unsigned __int8)sub_B2D610(v5, 47) )
    return 1;
  v6 = sub_B2D610(v5, 18);
  if ( v6 )
    return 1;
  if ( !a3 )
    return v6;
  v8 = sub_2F06CB0(a3, a1);
  v9 = *a3;
  v10 = v8;
  if ( *a3 == 0 || a2 == 0 )
    return v6;
  v11 = *(int **)(a2 + 8);
  if ( !v11 )
    return v6;
  v6 = qword_4F91668[8];
  if ( LOBYTE(qword_4F91668[8]) )
    return v6;
  v6 = qword_4F91BA8[8];
  if ( !LOBYTE(qword_4F91BA8[8]) )
    return v6;
  if ( LOBYTE(qword_4F919E8[8]) )
    goto LABEL_17;
  v12 = *v11;
  if ( !v12 )
  {
    if ( LOBYTE(qword_4F91908[8]) )
      goto LABEL_17;
    goto LABEL_13;
  }
  if ( v12 != 2 )
  {
LABEL_13:
    if ( !LOBYTE(qword_4F91AC8[8]) )
    {
LABEL_14:
      v13 = qword_4F91588[8];
      v14 = sub_2E3A020(v9, v10);
      if ( v15 )
        return !sub_D85370(a2, v13, (unsigned __int64)v14);
      return v6;
    }
LABEL_16:
    if ( !(unsigned __int8)sub_D84430(a2) )
      goto LABEL_17;
    goto LABEL_25;
  }
  if ( !(unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91828[8])
    || (unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91748[8]) )
  {
LABEL_17:
    v16 = sub_2E3A020(v9, v10);
    v6 = v17;
    if ( v17 )
      return sub_D84450(a2, (unsigned __int64)v16);
    return v6;
  }
  if ( LOBYTE(qword_4F91AC8[8]) )
    goto LABEL_16;
LABEL_25:
  v18 = *(_DWORD **)(a2 + 8);
  if ( !v18 || *v18 != 2 )
    goto LABEL_14;
  v19 = qword_4F914A8[8];
  v20 = sub_2E3A020(v9, v10);
  v6 = v21;
  if ( !v21 )
    return v6;
  return sub_D853A0(a2, v19, (unsigned __int64)v20);
}
