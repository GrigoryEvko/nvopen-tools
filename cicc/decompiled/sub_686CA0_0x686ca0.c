// Function: sub_686CA0
// Address: 0x686ca0
//
__int64 __fastcall sub_686CA0(unsigned int a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _DWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 result; // rax

  v6 = sub_67D610(a1, (_DWORD *)a2, 2u);
  if ( a3 )
  {
    v7 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    {
      a2 = 40;
      v7 = sub_823020((unsigned int)dword_4D03A00, 40);
    }
    else
    {
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    }
    *(_QWORD *)(v7 + 8) = 0;
    *(_DWORD *)v7 = 5;
    *(_QWORD *)(v7 + 16) = a3;
    if ( !*((_QWORD *)v6 + 23) )
      *((_QWORD *)v6 + 23) = v7;
    v8 = *((_QWORD *)v6 + 24);
    if ( v8 )
      *(_QWORD *)(v8 + 8) = v7;
    *((_QWORD *)v6 + 24) = v7;
  }
  if ( !a4 )
    return sub_6837D0((__int64)v6, (FILE *)a2);
  if ( !*a4 )
    *a4 = v6;
  result = a4[1];
  if ( result )
    *(_QWORD *)(result + 8) = v6;
  a4[1] = v6;
  return result;
}
