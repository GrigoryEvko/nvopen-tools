// Function: sub_8642D0
// Address: 0x8642d0
//
_QWORD *__fastcall sub_8642D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD *result; // rax
  __int64 v9; // rdx

  while ( 1 )
  {
    result = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    v9 = result[66];
    if ( v9 > 0 )
    {
      result[66] = v9 - 1;
      return result;
    }
    v6 = *(_QWORD *)(*(_QWORD *)(result[23] + 32LL) + 40LL);
    if ( !v6 || *(_BYTE *)(v6 + 28) != 3 )
      break;
    v7 = *(_QWORD *)(v6 + 32);
    result = sub_863FC0(a1, a2, v9, a4, a5, a6);
    if ( !v7 )
      return result;
  }
  return sub_863FC0(a1, a2, v9, a4, a5, a6);
}
