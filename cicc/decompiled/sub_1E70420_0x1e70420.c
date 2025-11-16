// Function: sub_1E70420
// Address: 0x1e70420
//
__int64 __fastcall sub_1E70420(__int64 a1, __int64 a2, __int64 a3)
{
  _DWORD *v4; // rsi
  __int64 result; // rax

  v4 = (_DWORD *)(*(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( ((*(_BYTE *)a3 ^ 6) & 6) != 0 || *(_DWORD *)(a3 + 8) <= 3u )
  {
    result = (unsigned int)(*(_DWORD *)(a2 + 248) + *(_DWORD *)(a3 + 12));
    if ( v4[62] < (unsigned int)result )
      v4[62] = result;
    if ( v4[52]-- == 1 )
    {
      result = a1 + 344;
      if ( v4 != (_DWORD *)(a1 + 344) )
        return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2120) + 120LL))(*(_QWORD *)(a1 + 2120));
    }
  }
  else
  {
    --v4[54];
    result = *(_QWORD *)a3 ^ 6LL;
    if ( (result & 6) == 0 && *(_DWORD *)(a3 + 8) == 5 )
      *(_QWORD *)(a1 + 2264) = v4;
  }
  return result;
}
