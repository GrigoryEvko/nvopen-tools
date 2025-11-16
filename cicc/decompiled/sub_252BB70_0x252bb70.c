// Function: sub_252BB70
// Address: 0x252bb70
//
__int64 __fastcall sub_252BB70(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax

  if ( !a4 )
    return 0;
  v5 = sub_250D2C0(a3, 0);
  v7 = sub_252B790(a1, v5, v6, a2, 1, 0, 1);
  if ( v7 )
    return *(unsigned __int8 *)(v7 + 97);
  else
    return 0;
}
