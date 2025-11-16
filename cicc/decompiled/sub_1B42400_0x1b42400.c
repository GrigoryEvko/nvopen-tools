// Function: sub_1B42400
// Address: 0x1b42400
//
__int64 __fastcall sub_1B42400(__int64 ***a1, __int64 a2)
{
  __int64 ***v2; // r12
  unsigned __int8 v3; // al
  __int64 **v5; // rax
  char v6; // dl

  v2 = a1;
  v3 = *((_BYTE *)a1 + 16);
  if ( v3 == 13 )
    return (__int64)v2;
  if ( v3 > 0x10u || *((_BYTE *)*a1 + 8) != 15 )
    return 0;
  v5 = (__int64 **)sub_15A9650(a2, (__int64)*a1);
  v6 = *((_BYTE *)a1 + 16);
  if ( v6 == 15 )
    return sub_159C470((__int64)v5, 0, 0);
  if ( v6 != 5 )
    return 0;
  if ( *((_WORD *)a1 + 9) != 46 )
    return 0;
  v2 = (__int64 ***)a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
  if ( *((_BYTE *)v2 + 16) != 13 )
    return 0;
  if ( v5 == *v2 )
    return (__int64)v2;
  return sub_15A4750(v2, v5, 0);
}
