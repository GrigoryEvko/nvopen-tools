// Function: sub_2252480
// Address: 0x2252480
//
__int64 __fastcall sub_2252480(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r8
  __int64 v6; // rdi

  v5 = (_QWORD *)((char *)a1 + *(_QWORD *)(*a1 - 16LL));
  v6 = *(_QWORD *)(*a1 - 8LL);
  if ( *(_QWORD *)(*v5 - 8LL) == v6 )
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD *, __int64, _QWORD *))(*(_QWORD *)v6 + 56LL))(
      v6,
      a4,
      6,
      a3,
      v5,
      a2,
      a1);
  return 0;
}
