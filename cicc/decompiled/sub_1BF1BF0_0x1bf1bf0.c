// Function: sub_1BF1BF0
// Address: 0x1bf1bf0
//
__int64 __fastcall sub_1BF1BF0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  int v4; // eax
  __int64 result; // rax

  *(_QWORD *)a1 = "vectorize.width";
  *(_DWORD *)(a1 + 24) = a3;
  v4 = dword_50524C8[0];
  *(_QWORD *)(a1 + 72) = a2;
  *(_QWORD *)(a1 + 80) = a4;
  *(_DWORD *)(a1 + 8) = v4;
  *(_QWORD *)(a1 + 16) = "interleave.count";
  *(_QWORD *)(a1 + 32) = "vectorize.enable";
  *(_QWORD *)(a1 + 40) = 0x2FFFFFFFFLL;
  *(_QWORD *)(a1 + 48) = "isvectorized";
  *(_QWORD *)(a1 + 56) = 0x300000000LL;
  *(_DWORD *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 28) = 1;
  *(_BYTE *)(a1 + 64) = 0;
  sub_1BF1A00(a1);
  result = sub_385D890();
  if ( (_BYTE)result )
  {
    result = dword_50523E8[0];
    *(_DWORD *)(a1 + 24) = dword_50523E8[0];
  }
  if ( *(_DWORD *)(a1 + 56) != 1 )
  {
    result = 0;
    if ( *(_DWORD *)(a1 + 8) == 1 )
      result = *(_DWORD *)(a1 + 24) == 1;
    *(_DWORD *)(a1 + 56) = result;
  }
  return result;
}
