// Function: sub_FC99C0
// Address: 0xfc99c0
//
__int64 __fastcall sub_FC99C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v7; // rax
  char v8; // dl

  if ( !a2 )
    return 0;
  v7 = sub_FC95E0(*a1, a2, a3, a4, a5, a6);
  if ( v8 )
    return (__int64)v7;
  if ( (*(_BYTE *)(a2 + 1) & 0x7F) == 1 )
    return sub_FC83A0((__int64)a1, a2);
  return v6;
}
