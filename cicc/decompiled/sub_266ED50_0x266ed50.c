// Function: sub_266ED50
// Address: 0x266ed50
//
__int64 __fastcall sub_266ED50(__int64 a1, __int64 a2, __int64 *a3)
{
  _DWORD *v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // r12d

  v5 = *(_DWORD **)a1;
  v6 = *a3;
  v7 = *(unsigned __int8 *)(*(_QWORD *)a1 + 241LL);
  if ( !(_BYTE)v7 )
  {
    v7 = 1;
    if ( !v6 )
      return v7;
LABEL_3:
    sub_250ED80(a2, (__int64)v5, v6, 1);
    return v7;
  }
  if ( !v5[72] || !v5[40] && !v5[56] )
  {
    if ( !v6 )
      return v7;
    goto LABEL_3;
  }
  return 0;
}
