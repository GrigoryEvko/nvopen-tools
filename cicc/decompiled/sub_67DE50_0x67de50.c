// Function: sub_67DE50
// Address: 0x67de50
//
_DWORD *__fastcall sub_67DE50(unsigned __int8 a1, unsigned int a2, _DWORD *a3, __int64 a4)
{
  _DWORD *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx

  v5 = sub_67D610(a2, a3, a1);
  v6 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v6 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_DWORD *)v6 = 4;
  *(_QWORD *)(v6 + 8) = 0;
  *(_QWORD *)(v6 + 24) = 0xFFFFFFFFLL;
  *(_WORD *)(v6 + 32) = 0;
  *(_BYTE *)(v6 + 34) = 0;
  *(_QWORD *)(v6 + 16) = a4;
  if ( !*((_QWORD *)v5 + 23) )
    *((_QWORD *)v5 + 23) = v6;
  v7 = *((_QWORD *)v5 + 24);
  if ( v7 )
    *(_QWORD *)(v7 + 8) = v6;
  *((_QWORD *)v5 + 24) = v6;
  return v5;
}
