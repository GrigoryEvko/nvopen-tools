// Function: sub_686E10
// Address: 0x686e10
//
__int64 __fastcall sub_686E10(unsigned int a1, FILE *a2, __int64 a3, _QWORD *a4)
{
  _DWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 result; // rax

  v6 = sub_67D610(a1, a2, 2u);
  if ( a3 )
  {
    v7 = sub_67BB20(4);
    *(_QWORD *)(v7 + 16) = a3;
    *(_DWORD *)(v7 + 24) = -1;
    if ( *((_QWORD *)v6 + 23) )
    {
      v8 = *((_QWORD *)v6 + 24);
      if ( !v8 )
      {
LABEL_5:
        *((_QWORD *)v6 + 24) = v7;
        goto LABEL_6;
      }
    }
    else
    {
      v8 = *((_QWORD *)v6 + 24);
      *((_QWORD *)v6 + 23) = v7;
      if ( !v8 )
        goto LABEL_5;
    }
    *(_QWORD *)(v8 + 8) = v7;
    goto LABEL_5;
  }
LABEL_6:
  if ( !a4 )
    return sub_6837D0((__int64)v6, a2);
  if ( !*a4 )
    *a4 = v6;
  result = a4[1];
  if ( result )
    *(_QWORD *)(result + 8) = v6;
  a4[1] = v6;
  return result;
}
