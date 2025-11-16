// Function: sub_303E700
// Address: 0x303e700
//
unsigned __int16 __fastcall sub_303E700(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rsi
  unsigned __int16 result; // ax

  if ( !a2 )
    return sub_AE5020(a5, a3);
  v8 = *(_QWORD *)(a2 - 32);
  if ( v8 && !*(_BYTE *)v8 && *(_QWORD *)(a2 + 80) == *(_QWORD *)(v8 + 24) )
    return sub_303E6A0(a1, v8, a3, a4, a5);
  if ( *(_BYTE *)a2 != 85 || (result = sub_CE94A0(a2, a4), !HIBYTE(result)) )
  {
    v8 = sub_307AA50(a2);
    if ( v8 )
      return sub_303E6A0(a1, v8, a3, a4, a5);
    return sub_AE5020(a5, a3);
  }
  return result;
}
