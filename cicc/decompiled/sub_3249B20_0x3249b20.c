// Function: sub_3249B20
// Address: 0x3249b20
//
__int64 __fastcall sub_3249B20(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 (__fastcall *v4)(__int64, unsigned int); // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 **v7; // rsi

  v3 = *(_QWORD **)(*(_QWORD *)(a1 + 16) + 184LL);
  v4 = *(__int64 (__fastcall **)(__int64, unsigned int))(*v3 + 480LL);
  if ( v4 == sub_31D48B0 )
  {
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v3[29] + 16LL) + 200LL))(*(_QWORD *)(v3[29] + 16LL));
    v6 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v5 + 16LL))(v5, (unsigned int)a2, 0);
  }
  else
  {
    v6 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))v4)(v3, a2, 0);
  }
  v7 = (unsigned __int64 **)(a1 + 120);
  if ( !*(_BYTE *)(a1 + 136) )
    v7 = *(unsigned __int64 ***)(a1 + 112);
  return sub_3249B00(*(__int64 **)(a1 + 16), v7, 15, v6);
}
