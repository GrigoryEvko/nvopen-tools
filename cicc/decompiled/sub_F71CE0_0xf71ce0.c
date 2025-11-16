// Function: sub_F71CE0
// Address: 0xf71ce0
//
__int64 __fastcall sub_F71CE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  if ( *(_DWORD *)(a2 + 8) )
    sub_F6B9F0(a1, (char **)a2, a3, a4, a5, a6);
  *(_QWORD *)(a1 + 64) = *(_QWORD *)(a2 + 64);
  *(_BYTE *)(a1 + 72) = *(_BYTE *)(a2 + 72);
  result = *(_QWORD *)(a2 + 80);
  *(_BYTE *)(a1 + 88) = 1;
  *(_QWORD *)(a1 + 80) = result;
  return result;
}
