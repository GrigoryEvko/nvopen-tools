// Function: sub_643D30
// Address: 0x643d30
//
__int64 __fastcall sub_643D30(__int64 a1)
{
  __int64 result; // rax

  result = *(_BYTE *)(a1 + 133) & 0x28;
  if ( (_BYTE)result == 32 )
  {
    sub_7BDC00();
    *(_BYTE *)(a1 + 133) &= ~0x20u;
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_BYTE *)(result + 12) &= ~8u;
  }
  return result;
}
