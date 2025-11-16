// Function: sub_6875A0
// Address: 0x6875a0
//
__int64 __fastcall sub_6875A0(unsigned int a1, FILE *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  _DWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax

  v8 = sub_67D610(a1, a2, 2u);
  if ( a3 )
  {
    v9 = sub_67BB20(4);
    *(_QWORD *)(v9 + 16) = a3;
    *(_DWORD *)(v9 + 24) = -1;
    if ( *((_QWORD *)v8 + 23) )
    {
      v10 = *((_QWORD *)v8 + 24);
      if ( !v10 )
      {
LABEL_5:
        *((_QWORD *)v8 + 24) = v9;
        goto LABEL_6;
      }
    }
    else
    {
      v10 = *((_QWORD *)v8 + 24);
      *((_QWORD *)v8 + 23) = v9;
      if ( !v10 )
        goto LABEL_5;
    }
    *(_QWORD *)(v10 + 8) = v9;
    goto LABEL_5;
  }
LABEL_6:
  if ( a4 )
  {
    a2 = (FILE *)a4;
    sub_67D780((__int64)v8, a4);
  }
  if ( !a5 )
    return sub_6837D0((__int64)v8, a2);
  if ( !*a5 )
    *a5 = v8;
  result = a5[1];
  if ( result )
    *(_QWORD *)(result + 8) = v8;
  a5[1] = v8;
  return result;
}
