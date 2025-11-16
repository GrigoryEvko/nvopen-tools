// Function: sub_2F29220
// Address: 0x2f29220
//
__int64 __fastcall sub_2F29220(__int64 a1, int a2, unsigned int a3)
{
  __int64 result; // rax

  result = 0;
  if ( *(_DWORD *)(a1 + 16) == 1 )
  {
    sub_2EAB0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL, a2);
    if ( a3 )
    {
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL * (unsigned int)(*(_DWORD *)(a1 + 16) + 1) + 24) = a3;
    }
    else
    {
      *(_DWORD *)(a1 + 16) = -1;
      sub_2E8A650(*(_QWORD *)(a1 + 8), 2u);
      sub_2E88D70(*(_QWORD *)(a1 + 8), (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) - 800LL));
    }
    return 1;
  }
  return result;
}
