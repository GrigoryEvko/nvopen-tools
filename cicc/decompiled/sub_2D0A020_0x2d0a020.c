// Function: sub_2D0A020
// Address: 0x2d0a020
//
__int64 __fastcall sub_2D0A020(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi

  if ( !(unsigned __int8)sub_2D04210(*a2) )
    return sub_2D09560(a1, (__int64)a2, a3, a4, a5);
  v12 = *a2;
  if ( *(_BYTE *)*a2 != 22 )
    v12 = 0;
  return sub_2D053E0(a1, v12, a5, v9, v10, v11);
}
