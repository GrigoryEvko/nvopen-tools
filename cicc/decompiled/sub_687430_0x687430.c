// Function: sub_687430
// Address: 0x687430
//
__int64 __fastcall sub_687430(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  _DWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 result; // rax

  v8 = sub_67D610(a1, (_DWORD *)a2, 2u);
  if ( a3 )
  {
    v9 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      a2 = 40;
      v9 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v9 + 8) = 0;
    *(_DWORD *)v9 = 5;
    *(_QWORD *)(v9 + 16) = a3;
    if ( !*((_QWORD *)v8 + 23) )
      *((_QWORD *)v8 + 23) = v9;
    v10 = *((_QWORD *)v8 + 24);
    if ( v10 )
      *(_QWORD *)(v10 + 8) = v9;
    *((_QWORD *)v8 + 24) = v9;
  }
  if ( a4 )
  {
    v11 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      a2 = 40;
      v11 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v11 + 8) = 0;
    *(_DWORD *)v11 = 5;
    *(_QWORD *)(v11 + 16) = a4;
    if ( !*((_QWORD *)v8 + 23) )
      *((_QWORD *)v8 + 23) = v11;
    v12 = *((_QWORD *)v8 + 24);
    if ( v12 )
      *(_QWORD *)(v12 + 8) = v11;
    *((_QWORD *)v8 + 24) = v11;
  }
  if ( !a5 )
    return sub_6837D0((__int64)v8, (FILE *)a2);
  if ( !*a5 )
    *a5 = v8;
  result = a5[1];
  if ( result )
    *(_QWORD *)(result + 8) = v8;
  a5[1] = v8;
  return result;
}
