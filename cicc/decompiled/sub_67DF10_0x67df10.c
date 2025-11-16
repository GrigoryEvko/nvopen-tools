// Function: sub_67DF10
// Address: 0x67df10
//
_DWORD *__fastcall sub_67DF10(unsigned __int8 a1, unsigned int a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  _DWORD *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx

  v7 = sub_67D610(a2, a3, a1);
  v8 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v8 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v8 + 8) = 0;
  *(_DWORD *)v8 = 3;
  *(_QWORD *)(v8 + 16) = a4;
  if ( !*((_QWORD *)v7 + 23) )
    *((_QWORD *)v7 + 23) = v8;
  v9 = *((_QWORD *)v7 + 24);
  if ( v9 )
    *(_QWORD *)(v9 + 8) = v8;
  *((_QWORD *)v7 + 24) = v8;
  v10 = sub_67BB20(4);
  *(_QWORD *)(v10 + 16) = a5;
  *(_DWORD *)(v10 + 24) = -1;
  if ( !*((_QWORD *)v7 + 23) )
    *((_QWORD *)v7 + 23) = v10;
  v11 = *((_QWORD *)v7 + 24);
  if ( v11 )
    *(_QWORD *)(v11 + 8) = v10;
  *((_QWORD *)v7 + 24) = v10;
  return v7;
}
