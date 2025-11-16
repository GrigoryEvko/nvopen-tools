// Function: sub_20DF6F0
// Address: 0x20df6f0
//
__int64 __fastcall sub_20DF6F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax

  v4 = sub_20DF640(a1, a2);
  return (*(__int64 (__fastcall **)(_QWORD, _QWORD, unsigned __int64))(**(_QWORD **)(a1 + 472) + 240LL))(
           *(_QWORD *)(a1 + 472),
           **(unsigned __int16 **)(a2 + 16),
           *(unsigned int *)(*(_QWORD *)(a1 + 232) + 8LL * *(int *)(a3 + 48)) - (unsigned __int64)v4);
}
