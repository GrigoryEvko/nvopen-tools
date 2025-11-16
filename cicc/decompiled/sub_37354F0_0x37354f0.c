// Function: sub_37354F0
// Address: 0x37354f0
//
__int64 __fastcall sub_37354F0(__int64 a1)
{
  int v1; // r13d
  int v2; // r12d

  v1 = 0;
  if ( (unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 4u )
    v1 = 8 * (*(_BYTE *)(*(_QWORD *)(a1 + 208) + 3769LL) != 0);
  v2 = sub_31DF6B0(*(_QWORD *)(a1 + 184));
  return (unsigned int)((unsigned __int16)sub_3220AA0(*(_QWORD *)(a1 + 208)) > 4u) + v1 + v2 + 3;
}
