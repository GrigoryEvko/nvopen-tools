// Function: sub_3218D50
// Address: 0x3218d50
//
__int64 __fastcall sub_3218D50(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdi
  __int64 v5; // r8
  void (*v6)(); // rax

  v4 = *(_QWORD **)(a1 + 8);
  v5 = v4[28];
  v6 = *(void (**)())(*(_QWORD *)v5 + 120LL);
  if ( v6 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __int64, __int64))v6)(v5, a3, 1);
    v4 = *(_QWORD **)(a1 + 8);
  }
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*v4 + 424LL))(v4, a2, 0);
}
