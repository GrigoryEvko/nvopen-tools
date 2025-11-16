// Function: sub_823970
// Address: 0x823970
//
__int64 __fastcall sub_823970(__int64 a1)
{
  __int64 v1; // rsi
  unsigned int i; // edx
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r8

  v1 = 1;
  if ( a1 )
    v1 = a1;
  if ( !qword_4F195D0 )
    return sub_822F50(0, v1);
  for ( i = v1 & *(_DWORD *)(qword_4F195D0 + 8); ; i = *(_DWORD *)(qword_4F195D0 + 8) & (i + 1) )
  {
    v3 = (_QWORD *)(*(_QWORD *)qword_4F195D0 + 16LL * i);
    if ( v1 == *v3 )
      break;
    if ( !*v3 )
      return sub_822F50(0, v1);
  }
  v4 = (_QWORD *)v3[1];
  if ( v4 && (v5 = v4[2], v5 > 0) && (v6 = v5 - 1, v7 = *(_QWORD *)(*v4 + 8 * v6), v4[2] = v6, v7) )
    return v7;
  else
    return sub_822F50(0, v1);
}
