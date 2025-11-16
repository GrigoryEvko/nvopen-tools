// Function: sub_2054060
// Address: 0x2054060
//
_QWORD *__fastcall sub_2054060(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 *a6, int a7)
{
  int v7; // eax
  __int64 v8; // rdi

  v7 = *(unsigned __int16 *)(a2 + 24);
  v8 = *(_QWORD *)(a1 + 552);
  if ( v7 == 14 || v7 == 36 )
    return (_QWORD *)sub_1D240D0(v8, a4, a5, *(_DWORD *)(a2 + 84), 0, a6, a7);
  else
    return sub_1D24380(v8, a4, a5, a2, a3, 0, a6, a7);
}
