// Function: sub_686A30
// Address: 0x686a30
//
__int64 __fastcall sub_686A30(unsigned __int8 a1, unsigned int a2, _DWORD *a3, _QWORD *a4, __int64 a5)
{
  __int64 v7; // rsi
  _DWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx

  v7 = (__int64)a3;
  v10 = sub_67D610(a2, a3, a1);
  if ( a5 )
  {
    v11 = sub_67BB20(4);
    *(_QWORD *)(v11 + 16) = a5;
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
  if ( a4 )
  {
    v13 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v7 = 40;
      v13 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_DWORD *)v13 = 2;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = *(_QWORD *)&dword_4F077C8;
    *(_QWORD *)(v13 + 16) = *a4;
    if ( !*((_QWORD *)v10 + 23) )
      *((_QWORD *)v10 + 23) = v13;
    v14 = *((_QWORD *)v10 + 24);
    if ( v14 )
      *(_QWORD *)(v14 + 8) = v13;
    *((_QWORD *)v10 + 24) = v13;
  }
  return sub_6837D0((__int64)v10, (FILE *)v7);
}
