// Function: sub_67E0D0
// Address: 0x67e0d0
//
_DWORD *__fastcall sub_67E0D0(unsigned int a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  _DWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx

  v6 = sub_67D610(a1, a2, 8u);
  v7 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v7 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v7 + 8) = 0;
  *(_DWORD *)v7 = 3;
  *(_QWORD *)(v7 + 16) = a3;
  if ( !*((_QWORD *)v6 + 23) )
    *((_QWORD *)v6 + 23) = v7;
  v8 = *((_QWORD *)v6 + 24);
  if ( v8 )
    *(_QWORD *)(v8 + 8) = v7;
  *((_QWORD *)v6 + 24) = v7;
  v9 = sub_67BB20(4);
  *(_QWORD *)(v9 + 16) = a4;
  *(_DWORD *)(v9 + 24) = -1;
  if ( !*((_QWORD *)v6 + 23) )
    *((_QWORD *)v6 + 23) = v9;
  v10 = *((_QWORD *)v6 + 24);
  if ( v10 )
    *(_QWORD *)(v10 + 8) = v9;
  *((_QWORD *)v6 + 24) = v9;
  return v6;
}
