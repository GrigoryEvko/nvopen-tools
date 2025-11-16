// Function: sub_35C8780
// Address: 0x35c8780
//
__int64 __fastcall sub_35C8780(_QWORD *a1, __int64 a2)
{
  _BYTE *v4; // rsi

  if ( (unsigned __int8)sub_2E799E0(a2)
    && *(_BYTE *)(a2 + 581)
    && (v4 = *(_BYTE **)a2, !(unsigned __int8)sub_BB98D0(a1, (__int64)v4)) )
  {
    return sub_35C7810((_QWORD *)a2, v4);
  }
  else
  {
    return 0;
  }
}
