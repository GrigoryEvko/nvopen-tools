// Function: sub_67E020
// Address: 0x67e020
//
_DWORD *__fastcall sub_67E020(unsigned int a1, _DWORD *a2, __int64 a3)
{
  _DWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx

  v4 = sub_67D610(a1, a2, 8u);
  v5 = qword_4D039F0;
  if ( !qword_4D039F0 || dword_4D03A00 == -1 )
    v5 = sub_823020((unsigned int)dword_4D03A00, 40);
  else
    qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
  *(_DWORD *)v5 = 4;
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)(v5 + 24) = 0xFFFFFFFFLL;
  *(_WORD *)(v5 + 32) = 0;
  *(_BYTE *)(v5 + 34) = 0;
  *(_QWORD *)(v5 + 16) = a3;
  if ( !*((_QWORD *)v4 + 23) )
    *((_QWORD *)v4 + 23) = v5;
  v6 = *((_QWORD *)v4 + 24);
  if ( v6 )
    *(_QWORD *)(v6 + 8) = v5;
  *((_QWORD *)v4 + 24) = v5;
  return v4;
}
