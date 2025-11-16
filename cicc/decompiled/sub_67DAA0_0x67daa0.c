// Function: sub_67DAA0
// Address: 0x67daa0
//
_DWORD *__fastcall sub_67DAA0(unsigned int a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  _DWORD *v6; // rax
  __int64 v7; // rdi
  _DWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx

  v6 = sub_67D610(a1, a2, 8u);
  v7 = (unsigned int)dword_4D03A00;
  v8 = v6;
  v9 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
  {
    v9 = sub_823020((unsigned int)dword_4D03A00, 40);
    v7 = (unsigned int)dword_4D03A00;
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
  v11 = qword_4D039F0;
  if ( !qword_4D039F0 || (_DWORD)v7 == -1 )
    v11 = sub_823020(v7, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v11 + 8) = 0;
  *(_DWORD *)v11 = 5;
  *(_QWORD *)(v11 + 16) = a4;
  if ( !*((_QWORD *)v8 + 23) )
    *((_QWORD *)v8 + 23) = v11;
  v12 = *((_QWORD *)v8 + 24);
  if ( v12 )
    *(_QWORD *)(v12 + 8) = v11;
  *((_QWORD *)v8 + 24) = v11;
  return v8;
}
