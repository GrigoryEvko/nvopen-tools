// Function: sub_3222A40
// Address: 0x3222a40
//
__int16 __fastcall sub_3222A40(__int64 a1)
{
  unsigned __int16 v1; // r8
  __int16 result; // ax

  v1 = sub_31DF670(*(_QWORD *)(a1 + 8));
  result = 23;
  if ( v1 <= 3u )
    return 6 - (!sub_31DF690(*(_QWORD *)(a1 + 8)) - 1);
  return result;
}
