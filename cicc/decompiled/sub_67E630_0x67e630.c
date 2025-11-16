// Function: sub_67E630
// Address: 0x67e630
//
__int64 __fastcall sub_67E630(unsigned int a1, _DWORD *a2, __int64 a3, int a4, _QWORD *a5)
{
  __int64 v7; // rbx
  _DWORD *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx

  v7 = a4;
  v8 = sub_67D610(a1, a2, 2u);
  v9 = sub_67BB20(4);
  *(_QWORD *)(v9 + 16) = a3;
  *(_DWORD *)(v9 + 24) = -1;
  if ( !*((_QWORD *)v8 + 23) )
    *((_QWORD *)v8 + 23) = v9;
  v10 = *((_QWORD *)v8 + 24);
  if ( v10 )
    *(_QWORD *)(v10 + 8) = v9;
  *((_QWORD *)v8 + 24) = v9;
  v11 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v11 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v11 + 8) = 0;
  *(_DWORD *)v11 = 0;
  *(_QWORD *)(v11 + 16) = v7;
  if ( !*((_QWORD *)v8 + 23) )
    *((_QWORD *)v8 + 23) = v11;
  v12 = *((_QWORD *)v8 + 24);
  if ( v12 )
    *(_QWORD *)(v12 + 8) = v11;
  *((_QWORD *)v8 + 24) = v11;
  return sub_67C730(a5, (__int64)v8);
}
