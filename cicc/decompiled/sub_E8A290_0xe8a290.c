// Function: sub_E8A290
// Address: 0xe8a290
//
__int64 __fastcall sub_E8A290(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // rbx

  v5 = a4;
  if ( !a4 )
    v5 = sub_E9A820();
  v6 = a1[1];
  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a1 + 176LL))(a1, *(_QWORD *)(*(_QWORD *)(v6 + 168) + 96LL), 0);
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD))(*a1 + 1312LL))(
           a1,
           0x7FFFFFFFFFFFFFFFLL,
           a3,
           v5,
           *(unsigned int *)(*(_QWORD *)(v6 + 152) + 8LL));
}
