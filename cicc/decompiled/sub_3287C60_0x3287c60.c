// Function: sub_3287C60
// Address: 0x3287c60
//
__int64 __fastcall sub_3287C60(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 112) + 37LL) & 0xF) == 0 )
    return (((unsigned __int8)*(_WORD *)(a1 + 32) >> 3) ^ 1) & 1;
  return result;
}
