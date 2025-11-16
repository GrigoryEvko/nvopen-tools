// Function: sub_DFDDE0
// Address: 0xdfdde0
//
__int64 __fastcall sub_DFDDE0(
        _QWORD *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 (__fastcall *v8)(__int64, _QWORD *, __int64, __int64, __int64, __int64, __int64, __int64); // rax

  v8 = *(__int64 (__fastcall **)(__int64, _QWORD *, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 1424LL);
  if ( v8 != sub_DF6370 )
    return ((__int64 (__fastcall *)(_QWORD, _QWORD *))v8)(*a1, a2);
  if ( BYTE4(a8) )
    return sub_BCD140(a2, 8 * (int)a8);
  return sub_BCB2B0(a2);
}
