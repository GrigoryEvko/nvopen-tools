// Function: sub_1695A10
// Address: 0x1695a10
//
__int64 __fastcall sub_1695A10(__int64 a1, char a2)
{
  __int64 v2; // rdx

  sub_1649960(a1);
  if ( v2 && sub_1695900(a1) && (!a2 || !(unsigned __int8)sub_15E3650(a1, 0)) )
    return ((((*(_BYTE *)(a1 + 32) & 0xF) + 15) & 0xFu) <= 2)
         | (unsigned int)((((*(_BYTE *)(a1 + 32) & 0xF) + 9) & 0xFu) <= 1);
  else
    return 0;
}
