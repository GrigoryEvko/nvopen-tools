// Function: sub_6E4430
// Address: 0x6e4430
//
__int64 __fastcall sub_6E4430(__int64 *a1, _QWORD *a2, _BYTE *a3, _DWORD *a4)
{
  __int64 v5; // r14
  __int64 v6; // r13
  bool v7; // zf
  __int64 result; // rax

  v5 = *a1;
  v6 = sub_6E3DA0(*a1, 0);
  v7 = *(_QWORD *)(v5 + 56) == 0;
  *a4 = *(_QWORD *)(v5 + 56) != 0;
  if ( !v7 )
    sub_6CC3B0(*(_QWORD *)(*(_QWORD *)(v5 + 56) + 8LL), (__int64)a1, a3);
  result = *(_QWORD *)(v6 + 356);
  *a2 = result;
  return result;
}
