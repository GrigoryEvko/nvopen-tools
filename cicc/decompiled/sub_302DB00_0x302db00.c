// Function: sub_302DB00
// Address: 0x302db00
//
__int64 __fastcall sub_302DB00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 200);
  if ( *(_QWORD *)(a2 + 48) != a2 + 40 && (*(_DWORD *)(v2 + 1624) <= 0x3Eu || *(_DWORD *)(v2 + 1628) <= 0x12Bu) )
    sub_C64ED0(".alias requires PTX version >= 6.3 and sm_30", 1u);
  result = sub_31E51E0(a1);
  *(_BYTE *)(a1 + 1096) = 0;
  return result;
}
