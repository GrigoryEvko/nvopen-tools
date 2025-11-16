// Function: sub_2FF53E0
// Address: 0x2ff53e0
//
__int64 __fastcall sub_2FF53E0(__int64 a1, int a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  __int16 v6; // dx
  bool v7; // zf
  __int64 result; // rax

  while ( 1 )
  {
    v5 = sub_2EBEE10(a3, a2);
    v6 = *(_WORD *)(v5 + 68);
    if ( v6 != 20 )
      break;
    a2 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL);
LABEL_7:
    if ( a2 >= 0 || !(unsigned __int8)sub_2EBEF70(a3, a2) )
      return 0;
  }
  if ( v6 == 12 )
  {
    a2 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 88LL);
    goto LABEL_7;
  }
  v7 = (unsigned __int8)sub_2EBEF70(a3, a2) == 0;
  result = 0;
  if ( !v7 )
    return (unsigned int)a2;
  return result;
}
