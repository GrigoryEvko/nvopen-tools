// Function: sub_ED2C10
// Address: 0xed2c10
//
__int64 __fastcall sub_ED2C10(__int64 a1, char a2)
{
  __int64 v2; // rdx

  sub_BD5D20(a1);
  if ( v2 && sub_ED2B40(a1, *(_QWORD *)(a1 + 40)) && (!a2 || !(unsigned __int8)sub_B2DDD0(a1, 0, 0, 1, 0, 0, 0)) )
    return ((((*(_BYTE *)(a1 + 32) & 0xF) + 9) & 0xFu) <= 1)
         | (unsigned int)((((*(_BYTE *)(a1 + 32) & 0xF) + 15) & 0xFu) <= 2);
  else
    return 0;
}
