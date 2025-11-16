// Function: sub_1F353B0
// Address: 0x1f353b0
//
__int64 __fastcall sub_1F353B0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 i; // rsi

  if ( a3 )
    (*(void (__fastcall **)(_QWORD))a3)(*(_QWORD *)(a3 + 8));
  for ( i = a2[12]; i != a2[11]; i = a2[12] )
    sub_1DD9130((__int64)a2, (__int64 *)(i - 8), 0);
  return sub_1DD6E70(a2);
}
