// Function: sub_6870E0
// Address: 0x6870e0
//
__int64 __fastcall sub_6870E0(unsigned __int8 a1, unsigned int a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  _DWORD *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx

  v8 = (__int64)a3;
  v12 = sub_67D610(a2, a3, a1);
  if ( a4 )
  {
    v13 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v8 = 40;
      v13 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v13 + 8) = 0;
    *(_DWORD *)v13 = 3;
    *(_QWORD *)(v13 + 16) = a4;
    if ( !*((_QWORD *)v12 + 23) )
      *((_QWORD *)v12 + 23) = v13;
    v14 = *((_QWORD *)v12 + 24);
    if ( v14 )
      *(_QWORD *)(v14 + 8) = v13;
    *((_QWORD *)v12 + 24) = v13;
  }
  if ( a5 )
  {
    v15 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      v8 = 40;
      v15 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v15 + 8) = 0;
    *(_DWORD *)v15 = 3;
    *(_QWORD *)(v15 + 16) = a5;
    if ( !*((_QWORD *)v12 + 23) )
      *((_QWORD *)v12 + 23) = v15;
    v16 = *((_QWORD *)v12 + 24);
    if ( v16 )
      *(_QWORD *)(v16 + 8) = v15;
    *((_QWORD *)v12 + 24) = v15;
  }
  if ( a6 )
  {
    v17 = sub_67BB20(4);
    *(_QWORD *)(v17 + 16) = a6;
    *(_DWORD *)(v17 + 24) = -1;
    if ( *((_QWORD *)v12 + 23) )
    {
      v18 = *((_QWORD *)v12 + 24);
      if ( !v18 )
      {
LABEL_23:
        *((_QWORD *)v12 + 24) = v17;
        return sub_6837D0((__int64)v12, (FILE *)v8);
      }
    }
    else
    {
      v18 = *((_QWORD *)v12 + 24);
      *((_QWORD *)v12 + 23) = v17;
      if ( !v18 )
        goto LABEL_23;
    }
    *(_QWORD *)(v18 + 8) = v17;
    goto LABEL_23;
  }
  return sub_6837D0((__int64)v12, (FILE *)v8);
}
