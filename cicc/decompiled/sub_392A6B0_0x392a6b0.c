// Function: sub_392A6B0
// Address: 0x392a6b0
//
bool __fastcall sub_392A6B0(__int64 a1, __int64 a2)
{
  bool v2; // zf
  bool result; // al

  sub_39092C0(a1);
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  v2 = *(_QWORD *)(a2 + 56) == 0;
  *(_QWORD *)(a1 + 136) = a2;
  *(_QWORD *)a1 = &unk_4A3ED18;
  result = 1;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 168) = 257;
  if ( !v2 )
    result = **(_BYTE **)(a2 + 48) != 64;
  *(_BYTE *)(a1 + 113) = result;
  return result;
}
