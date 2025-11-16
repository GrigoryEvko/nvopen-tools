// Function: sub_5D1850
// Address: 0x5d1850
//
int __fastcall sub_5D1850(char *a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  _QWORD *v6; // r12
  __int64 v7; // rax
  _QWORD *v8; // rbx
  char *v9; // r12
  int result; // eax
  bool v11; // dl

  v5 = sub_5C7880(a1, a3);
  if ( !v5 )
    return 0;
  v6 = (_QWORD *)v5;
  v7 = qword_4CF6E10;
  if ( !qword_4CF6E10 )
  {
    v7 = sub_727670();
    qword_4CF6E10 = v7;
  }
  *(_BYTE *)(v7 + 9) = a3;
  *(_QWORD *)(v7 + 16) = a1;
  *(_QWORD *)(v7 + 24) = a2;
  v8 = (_QWORD *)*v6;
  if ( !*v6 )
    return 0;
  while ( 2 )
  {
    v9 = (char *)((**(_BYTE **)(v8[1] + 16LL) == 49) + *(_QWORD *)(v8[1] + 16LL));
    switch ( a3 )
    {
      case 0:
      case 5:
        if ( !sub_5CAB70(v9, qword_4CF6E10) && !sub_5CAA00(v9, qword_4CF6E10) )
        {
          a1 = v9;
          v11 = (unsigned int)sub_5C97E0(v9) != 0;
          result = v11;
LABEL_10:
          v8 = (_QWORD *)*v8;
          if ( !v8 || v11 )
            return result;
          continue;
        }
        return 1;
      case 1:
        a1 = (char *)((**(_BYTE **)(v8[1] + 16LL) == 49) + *(_QWORD *)(v8[1] + 16LL));
        result = sub_5CAB70(v9, qword_4CF6E10);
        v11 = result != 0;
        goto LABEL_10;
      case 2:
        a1 = (char *)((**(_BYTE **)(v8[1] + 16LL) == 49) + *(_QWORD *)(v8[1] + 16LL));
        result = sub_5CAA00(v9, qword_4CF6E10);
        v11 = result != 0;
        goto LABEL_10;
      case 3:
        a1 = (char *)((**(_BYTE **)(v8[1] + 16LL) == 49) + *(_QWORD *)(v8[1] + 16LL));
        result = sub_5C97E0(v9);
        v11 = result != 0;
        goto LABEL_10;
      default:
        sub_721090(a1);
    }
  }
}
