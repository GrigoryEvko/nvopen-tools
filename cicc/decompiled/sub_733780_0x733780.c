// Function: sub_733780
// Address: 0x733780
//
void __fastcall sub_733780(unsigned __int8 a1, __int64 a2, char *a3, char a4, int a5)
{
  int v5; // r12d
  char v6; // cl
  char *v7; // rbx
  int v8; // edx
  __int64 v9; // rax
  char *v11; // rax

  v5 = a5;
  if ( a3 )
  {
    v6 = *a3;
    v7 = a3;
    if ( !*a3 )
      goto LABEL_3;
    v8 = a5;
  }
  else
  {
    v11 = sub_726C30(a4);
    v6 = a4;
    v7 = v11;
    if ( !a4 )
    {
      if ( !a2 )
        goto LABEL_5;
      goto LABEL_23;
    }
    v8 = 0;
  }
  if ( !v5 || (v5 = v8, !*((_QWORD *)v7 + 4)) )
  {
    v9 = qword_4F06BC0;
    if ( v6 == 1 && a1 == 23 && (*(v7 - 8) & 1) != 0 && (*(_BYTE *)(qword_4F06BC0 - 8LL) & 1) == 0 )
      v9 = *(_QWORD *)(qword_4F04C68[0] + 488LL);
    *((_QWORD *)v7 + 4) = v9;
    if ( a2 && a1 == 23 )
    {
      if ( *(_BYTE *)(a2 + 28) == 17 )
      {
        *(_BYTE *)(v9 + 1) |= 2u;
        v5 = v8;
      }
      else
      {
        v5 = v8;
        if ( ((*(_BYTE *)(v9 - 8) ^ (unsigned __int8)*(v7 - 8)) & 1) == 0 )
        {
LABEL_18:
          *((_QWORD *)v7 + 7) = *(_QWORD *)(v9 + 48);
          *(_QWORD *)(v9 + 48) = v7;
          if ( v6 == 2 )
            *(_BYTE *)(v9 + 1) |= 1u;
          v5 = v8;
          *((_QWORD *)v7 + 5) = *(_QWORD *)(v9 + 24);
          goto LABEL_3;
        }
      }
LABEL_4:
      if ( v5 )
        goto LABEL_5;
LABEL_23:
      sub_732E60((unsigned __int8 *)v7, a1, (_QWORD *)a2);
      goto LABEL_5;
    }
    v5 = v8;
    if ( ((*(_BYTE *)(v9 - 8) ^ (unsigned __int8)*(v7 - 8)) & 1) != 0 )
      goto LABEL_3;
    goto LABEL_18;
  }
LABEL_3:
  if ( a2 )
    goto LABEL_4;
LABEL_5:
  qword_4F06BC0 = v7;
}
