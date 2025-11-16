// Function: sub_7D8E20
// Address: 0x7d8e20
//
__int64 __fastcall sub_7D8E20(__int64 a1)
{
  int v1; // r14d
  __int16 v2; // r13
  __int64 result; // rax

  v1 = dword_4F07508[0];
  v2 = dword_4F07508[1];
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 64);
  sub_7E2D70();
  sub_7D8DC0(*(_BYTE *)(a1 + 177), (__m128i **)(a1 + 184));
  result = dword_4F0696C;
  if ( dword_4F0696C )
  {
    result = *(unsigned __int8 *)(a1 + 169);
    if ( ((result & 1) != 0 || unk_4F072F1 && (result & 2) == 0) && !*(_BYTE *)(a1 + 136) && !*(_BYTE *)(a1 + 177) )
      *(_BYTE *)(a1 + 177) = 3;
  }
  dword_4F07508[0] = v1;
  LOWORD(dword_4F07508[1]) = v2;
  return result;
}
