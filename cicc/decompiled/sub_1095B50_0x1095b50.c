// Function: sub_1095B50
// Address: 0x1095b50
//
__int64 __fastcall sub_1095B50(__int64 a1, __int64 a2)
{
  bool v2; // zf
  bool v3; // al
  __int64 result; // rax

  sub_ECD550(a1);
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  v2 = *(_QWORD *)(a2 + 56) == 0;
  *(_QWORD *)(a1 + 144) = a2;
  *(_QWORD *)a1 = &unk_49E6290;
  v3 = 1;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 176) = 16777473;
  if ( !v2 )
    v3 = **(_BYTE **)(a2 + 48) != 64;
  *(_BYTE *)(a1 + 113) = v3;
  result = *(unsigned __int8 *)(a2 + 401);
  *(_BYTE *)(a1 + 119) = result;
  return result;
}
