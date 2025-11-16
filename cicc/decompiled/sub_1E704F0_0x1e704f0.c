// Function: sub_1E704F0
// Address: 0x1e704f0
//
__int64 __fastcall sub_1E704F0(__int64 a1, __int64 a2, __int64 a3)
{
  _DWORD *v4; // rsi
  __int64 result; // rax

  v4 = (_DWORD *)(*(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( ((*(_BYTE *)a3 ^ 6) & 6) != 0 || *(_DWORD *)(a3 + 8) <= 3u )
  {
    result = (unsigned int)(*(_DWORD *)(a2 + 252) + *(_DWORD *)(a3 + 12));
    if ( v4[63] < (unsigned int)result )
      v4[63] = result;
    if ( v4[53]-- == 1 )
    {
      result = a1 + 72;
      if ( v4 != (_DWORD *)(a1 + 72) )
        return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2120) + 128LL))(*(_QWORD *)(a1 + 2120));
    }
  }
  else
  {
    --v4[55];
    result = *(_QWORD *)a3 ^ 6LL;
    if ( (result & 6) == 0 && *(_DWORD *)(a3 + 8) == 5 )
      *(_QWORD *)(a1 + 2256) = v4;
  }
  return result;
}
