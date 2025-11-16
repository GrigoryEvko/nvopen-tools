// Function: sub_1097ED0
// Address: 0x1097ed0
//
__int64 __fastcall sub_1097ED0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5, unsigned int a6)
{
  const char *v6; // rsi
  bool v7; // al

  v6 = *(const char **)(a1 + 152);
  *(_QWORD *)(a1 + 104) = v6;
  while ( !(unsigned __int8)sub_1097E30(a1, v6, (__int64)a3, a4, a5, a6) )
  {
    v7 = sub_1097E90(a1, *(const char **)(a1 + 152));
    a3 = *(_BYTE **)(a1 + 152);
    if ( v7 || *a3 == 13 || *a3 == 10 || a3 == (_BYTE *)(*(_QWORD *)(a1 + 160) + *(_QWORD *)(a1 + 168)) )
      break;
    v6 = a3 + 1;
    *(_QWORD *)(a1 + 152) = a3 + 1;
  }
  return *(_QWORD *)(a1 + 104);
}
