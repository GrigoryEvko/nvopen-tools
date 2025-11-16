// Function: sub_686B60
// Address: 0x686b60
//
__int64 __fastcall sub_686B60(unsigned __int8 a1, unsigned int a2, FILE *a3, __int64 a4, __int64 a5)
{
  _DWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx

  v10 = sub_67D610(a2, a3, a1);
  if ( a4 )
  {
    v11 = sub_67BB20(4);
    *(_QWORD *)(v11 + 16) = a4;
    *(_DWORD *)(v11 + 24) = -1;
    if ( *((_QWORD *)v10 + 23) )
    {
      v12 = *((_QWORD *)v10 + 24);
      if ( !v12 )
      {
LABEL_5:
        *((_QWORD *)v10 + 24) = v11;
        goto LABEL_6;
      }
    }
    else
    {
      v12 = *((_QWORD *)v10 + 24);
      *((_QWORD *)v10 + 23) = v11;
      if ( !v12 )
        goto LABEL_5;
    }
    *(_QWORD *)(v12 + 8) = v11;
    goto LABEL_5;
  }
LABEL_6:
  if ( a5 )
  {
    v13 = sub_67BB20(4);
    *(_QWORD *)(v13 + 16) = a5;
    *(_DWORD *)(v13 + 24) = -1;
    if ( *((_QWORD *)v10 + 23) )
    {
      v14 = *((_QWORD *)v10 + 24);
      if ( !v14 )
      {
LABEL_10:
        *((_QWORD *)v10 + 24) = v13;
        return sub_6837D0((__int64)v10, a3);
      }
    }
    else
    {
      v14 = *((_QWORD *)v10 + 24);
      *((_QWORD *)v10 + 23) = v13;
      if ( !v14 )
        goto LABEL_10;
    }
    *(_QWORD *)(v14 + 8) = v13;
    goto LABEL_10;
  }
  return sub_6837D0((__int64)v10, a3);
}
