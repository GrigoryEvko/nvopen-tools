// Function: sub_7E9260
// Address: 0x7e9260
//
__int64 __fastcall sub_7E9260(__int64 a1, __int64 a2, _DWORD *a3)
{
  char v4; // dl
  __int64 result; // rax
  char v6; // al
  int v7; // eax

  v4 = *(_BYTE *)(a2 + 49);
  if ( (v4 & 1) != 0
    || dword_4D044B4
    || (v6 = *(_BYTE *)(a2 + 48), ((v6 - 6) & 0xFD) == 0)
    || (unsigned __int8)(v6 - 1) <= 1u
    || v6 == 5 && (v7 = sub_7F9D00(a2), v4 = *(_BYTE *)(a2 + 49), v7)
    || (v4 & 0x10) == 0 )
  {
    result = (__int64)sub_7E7C20(a1, 0, v4 & 1, 1);
    *a3 = 0;
  }
  else
  {
    result = sub_7E7CB0(a1);
    *a3 = 1;
  }
  return result;
}
