// Function: sub_6ECAE0
// Address: 0x6ecae0
//
__int64 __fastcall sub_6ECAE0(__int64 a1, int a2, int a3, int a4, unsigned __int8 a5, __int64 *a6, __int64 *a7)
{
  __int64 v11; // rsi
  __int64 v12; // r14

  v11 = sub_6EB460(a5, a1, a6);
  *a7 = v11;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u
    && *(_QWORD *)(v11 + 16)
    && !word_4D04898
    && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
  {
    if ( (unsigned int)sub_6E5430() )
      sub_6851C0(0x1Cu, a6);
    *(_QWORD *)(*a7 + 16) = 0;
    v11 = *a7;
  }
  v12 = sub_6EC670(a1, v11, a2, a3);
  if ( !a4 )
    sub_6EB560(a1, (__int64)a6);
  return v12;
}
