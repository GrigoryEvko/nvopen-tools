// Function: sub_2534E00
// Address: 0x2534e00
//
__int64 __fastcall sub_2534E00(__int64 a1)
{
  __int64 result; // rax
  bool v2; // zf

  result = *(unsigned __int8 *)(a1 + 96);
  v2 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_BYTE *)(a1 + 97) = result;
  if ( v2 )
    *(_BYTE *)(a1 + 112) = 1;
  return result;
}
