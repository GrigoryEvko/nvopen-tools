// Function: sub_34F1190
// Address: 0x34f1190
//
__int64 __fastcall sub_34F1190(__int64 a1, __int64 *a2)
{
  if ( (unsigned __int8)sub_BB98D0((_QWORD *)a1, *a2)
    || *(_QWORD *)(a1 + 648) && !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(a1 + 656))(a1 + 632, a2) )
  {
    return 0;
  }
  else
  {
    return sub_34ED530((__m128i *)a1, a2);
  }
}
