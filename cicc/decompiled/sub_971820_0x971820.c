// Function: sub_971820
// Address: 0x971820
//
__int64 __fastcall sub_971820(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  unsigned __int8 **v6; // rax
  unsigned __int8 **v7; // rbx
  __int64 result; // rax

  v6 = (unsigned __int8 **)sub_98ACB0(a1, 6);
  if ( *(_BYTE *)v6 != 3 )
    return 0;
  v7 = v6;
  if ( ((_BYTE)v6[10] & 1) == 0
    || (unsigned __int8)sub_B2FC80(v6)
    || (unsigned __int8)sub_B2F6B0(v7)
    || ((_BYTE)v7[10] & 2) != 0 )
  {
    return 0;
  }
  if ( v7 != (unsigned __int8 **)sub_BD45C0(a1, (_DWORD)a4, a3, 1, 0, 0, 0, 0) )
    return sub_96E500(*(v7 - 4), a2, (__int64)a4);
  result = sub_9714E0((__int64)*(v7 - 4), a2, a3, a4);
  if ( !result )
    return sub_96E500(*(v7 - 4), a2, (__int64)a4);
  return result;
}
