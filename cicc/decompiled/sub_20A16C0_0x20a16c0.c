// Function: sub_20A16C0
// Address: 0x20a16c0
//
unsigned __int64 __fastcall sub_20A16C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rdx

  result = sub_1D1FC50(a6, a2);
  if ( (_DWORD)result )
  {
    _BitScanReverse((unsigned int *)&result, result);
    result = (unsigned int)result ^ 0x1F;
    if ( (_DWORD)result != 31 )
    {
      v8 = *(_QWORD *)a4;
      result = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)result + 33);
      if ( *(_DWORD *)(a4 + 8) > 0x40u )
      {
        *(_QWORD *)v8 |= result;
      }
      else
      {
        result |= v8;
        *(_QWORD *)a4 = result;
      }
    }
  }
  return result;
}
