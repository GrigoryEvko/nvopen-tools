// Function: sub_335C0F0
// Address: 0x335c0f0
//
__int64 __fastcall sub_335C0F0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 result; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  int v12; // eax

  result = *(_QWORD *)(*a1 + 112LL);
  if ( (__int64 (*)())result == sub_334CAA0 || (result = ((__int64 (*)(void))result)(), !(_BYTE)result) )
  {
    if ( (*(_BYTE *)a5 & 6) == 0 )
    {
      v10 = (_QWORD *)a1[2];
      v11 = *(unsigned int *)(*(_QWORD *)(a3 + 40) + 40LL * a4 + 8);
      v12 = *(_DWORD *)(a3 + 24);
      if ( v12 < 0 )
        a4 += *(unsigned __int8 *)(v10[1] - 40LL * (unsigned int)~v12 + 4);
      result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64, __int64, __int64, _QWORD))(*v10 + 1112LL))(
                 v10,
                 a1[75],
                 a2,
                 v11,
                 a3,
                 a4);
      if ( (unsigned int)result > 1 )
      {
        if ( !BYTE4(result) )
          return result;
        if ( *(_DWORD *)(a3 + 24) == 49 )
        {
          if ( *(_DWORD *)(a1[73] + 120LL) )
            result = (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 40LL) + 96LL) < 0x80000000)
                   + (unsigned int)result
                   - 1;
        }
LABEL_11:
        *(_DWORD *)(a5 + 12) = result;
        return result;
      }
      if ( BYTE4(result) )
        goto LABEL_11;
    }
  }
  return result;
}
