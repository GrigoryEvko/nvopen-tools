// Function: sub_1F3AD60
// Address: 0x1f3ad60
//
__int64 __fastcall sub_1F3AD60(_QWORD a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5)
{
  __int16 *v5; // rax
  __int64 v6; // rdx

  if ( a3 < *(unsigned __int16 *)(a2 + 2) )
  {
    v5 = (__int16 *)(*(_QWORD *)(a2 + 40) + 8LL * a3);
    v6 = *v5;
    if ( (v5[1] & 1) != 0 )
      return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a4 + 144LL))(a4, a5, v6);
    if ( (v6 & 0x8000u) == 0LL )
      return *(_QWORD *)(a4[32] + 8 * v6);
  }
  return 0;
}
