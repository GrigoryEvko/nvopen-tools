// Function: sub_E71DA0
// Address: 0xe71da0
//
__int64 __fastcall sub_E71DA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rbx

  v5 = a3;
  v6 = a2;
  v7 = a1[1];
  if ( *(_BYTE *)(*(_QWORD *)(v7 + 152) + 280LL) )
  {
    v8 = sub_E6C430(a1[1], a2, a3, a4, a5);
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 272))(a1, v8, a2);
    v6 = sub_E808D0(v8, 0, v7, 0);
  }
  return sub_E9A5B0(a1, v6, v5, 0);
}
