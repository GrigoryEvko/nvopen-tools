// Function: sub_2EC6220
// Address: 0x2ec6220
//
__int64 __fastcall sub_2EC6220(__int64 a1, __int64 a2, __int64 a3)
{
  _DWORD *v4; // rsi
  __int64 result; // rax

  v4 = (_DWORD *)(*(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( ((*(_BYTE *)a3 ^ 6) & 6) != 0 || *(_DWORD *)(a3 + 8) <= 3u )
  {
    result = (unsigned int)(*(_DWORD *)(a2 + 236) + *(_DWORD *)(a3 + 12));
    if ( v4[59] < (unsigned int)result )
      v4[59] = result;
    if ( v4[55]-- == 1 )
    {
      result = a1 + 72;
      if ( v4 != (_DWORD *)(a1 + 72) )
        return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 3472) + 136LL))(*(_QWORD *)(a1 + 3472));
    }
  }
  else
  {
    --v4[57];
    result = *(_QWORD *)a3 ^ 6LL;
    if ( (result & 6) == 0 && *(_DWORD *)(a3 + 8) == 5 )
      *(_QWORD *)(a1 + 3520) = v4;
  }
  return result;
}
