// Function: sub_25BCCE0
// Address: 0x25bcce0
//
__int64 __fastcall sub_25BCCE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  _BYTE *v6; // r12

  v4 = *a1;
  if ( !*(_QWORD *)(a2 + 16) )
    sub_4263D6(a1, v4, a3);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64))(a2 + 24))(a2, v4) )
    return 0;
  if ( sub_B2FC80(*a1) )
    return 1;
  if ( !*(_BYTE *)(a2 + 100) )
    return 0;
  v6 = (_BYTE *)*a1;
  if ( sub_B2FC80(*a1) )
    return 1;
  else
    return sub_B2FC00(v6);
}
