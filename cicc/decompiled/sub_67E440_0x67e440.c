// Function: sub_67E440
// Address: 0x67e440
//
__int64 __fastcall sub_67E440(unsigned int a1, _DWORD *a2, int a3, _QWORD *a4)
{
  __int64 v5; // rbx
  _DWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx

  v5 = a3;
  v6 = sub_67D610(a1, a2, 2u);
  v7 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v7 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_QWORD *)(v7 + 8) = 0;
  *(_DWORD *)v7 = 0;
  *(_QWORD *)(v7 + 16) = v5;
  if ( !*((_QWORD *)v6 + 23) )
    *((_QWORD *)v6 + 23) = v7;
  v8 = *((_QWORD *)v6 + 24);
  if ( v8 )
    *(_QWORD *)(v8 + 8) = v7;
  *((_QWORD *)v6 + 24) = v7;
  return sub_67C730(a4, (__int64)v6);
}
