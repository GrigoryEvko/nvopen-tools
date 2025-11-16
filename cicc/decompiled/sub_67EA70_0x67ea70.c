// Function: sub_67EA70
// Address: 0x67ea70
//
_DWORD *__fastcall sub_67EA70(unsigned int a1, __int64 a2)
{
  _DWORD *v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx

  dword_4F07508[0] = 0;
  LOWORD(dword_4F07508[1]) = 1;
  v2 = sub_67D610(a1, dword_4F07508, 0xAu);
  if ( a2 )
  {
    v3 = sub_724840((unsigned int)dword_4D03A00, a2);
    v4 = qword_4D039F0;
    if ( !qword_4D039F0 || dword_4D03A00 == -1 )
      v4 = sub_823020((unsigned int)dword_4D03A00, 40);
    else
      qword_4D039F0 = *(_QWORD *)(qword_4D039F0 + 8);
    *(_QWORD *)(v4 + 8) = 0;
    *(_DWORD *)v4 = 3;
    *(_QWORD *)(v4 + 16) = v3;
    if ( !*((_QWORD *)v2 + 23) )
      *((_QWORD *)v2 + 23) = v4;
    v5 = *((_QWORD *)v2 + 24);
    if ( v5 )
      *(_QWORD *)(v5 + 8) = v4;
    *((_QWORD *)v2 + 24) = v4;
  }
  return v2;
}
