// Function: sub_106E470
// Address: 0x106e470
//
unsigned __int64 __fastcall sub_106E470(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  _QWORD *v4; // r13
  unsigned __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r14
  unsigned __int64 v12; // rax

  v4 = (_QWORD *)a4[1];
  if ( (a3 & 0x10) == 0 )
    return sub_E808D0(a2, 0, (_QWORD *)a4[1], 0);
  v7 = sub_E808D0(a2, 0, (_QWORD *)a4[1], 0);
  v11 = sub_E6C430((__int64)v4, 0, v8, v9, v10);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a4 + 208LL))(a4, v11, 0);
  v12 = sub_E808D0(v11, 0, v4, 0);
  return sub_E81A00(18, v7, v12, v4, 0);
}
