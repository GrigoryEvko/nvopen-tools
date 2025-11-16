// Function: sub_1D0B780
// Address: 0x1d0b780
//
__int64 __fastcall sub_1D0B780(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 result; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx

  result = *(_QWORD *)(*a1 + 104LL);
  if ( (__int64 (*)())result == sub_1CFBF50 || (result = ((__int64 (*)(void))result)(), !(_BYTE)result) )
  {
    if ( (*(_BYTE *)a5 & 6) == 0 )
    {
      v10 = (_QWORD *)a1[2];
      v11 = *(unsigned int *)(*(_QWORD *)(a3 + 32) + 40LL * a4 + 8);
      if ( *(__int16 *)(a3 + 24) < 0 )
        a4 += *(unsigned __int8 *)(v10[1] + ((__int64)~*(__int16 *)(a3 + 24) << 6) + 4);
      result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64, __int64, __int64, _QWORD))(*v10 + 832LL))(
                 v10,
                 a1[79],
                 a2,
                 v11,
                 a3,
                 a4);
      if ( (int)result <= 1 )
      {
        if ( (int)result < 0 )
          return result;
      }
      else if ( *(_WORD *)(a3 + 24) == 46 && *(_QWORD *)(a1[77] + 96LL) != *(_QWORD *)(a1[77] + 88LL) )
      {
        result = (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 32) + 40LL) + 84LL) < 0x80000000) + (unsigned int)result - 1;
      }
      *(_DWORD *)(a5 + 12) = result;
    }
  }
  return result;
}
