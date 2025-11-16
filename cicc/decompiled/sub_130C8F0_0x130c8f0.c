// Function: sub_130C8F0
// Address: 0x130c8f0
//
__int64 __fastcall sub_130C8F0(__int64 a1, __int64 a2)
{
  __int64 i; // r12
  __int64 result; // rax

  for ( i = sub_131C0E0(*(_QWORD *)(a2 + 58376)); ; sub_13446B0(a1, a2, i, result) )
  {
    result = sub_13441C0(a1, a2, i, a2 + 38936, 0);
    if ( !result )
      break;
  }
  return result;
}
