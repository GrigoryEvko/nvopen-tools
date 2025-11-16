// Function: sub_32421D0
// Address: 0x32421d0
//
__int64 __fastcall sub_32421D0(__int64 a1, __int64 a2, __int64 a3)
{
  bool v4; // zf
  __int64 v5; // rax
  void (__fastcall *v7)(__int64, __int64, _QWORD); // rax

  v4 = !sub_32420F0(a1);
  v5 = *(_QWORD *)a1;
  if ( v4 )
  {
    v7 = *(void (__fastcall **)(__int64, __int64, _QWORD))(v5 + 8);
    if ( a2 > 31 )
    {
      v7(a1, 146, 0);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 24LL))(a1, a2);
    }
    else
    {
      v7(a1, (unsigned __int8)(a2 + 112), 0);
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(v5 + 8))(a1, 146, 0);
    (**(void (__fastcall ***)(__int64, _QWORD))a1)(a1, (unsigned int)a2);
  }
  return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 16LL))(a1, a3);
}
