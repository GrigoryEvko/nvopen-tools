// Function: sub_3242120
// Address: 0x3242120
//
__int64 __fastcall sub_3242120(_BYTE *a1, __int64 a2, __int64 a3)
{
  bool v4; // zf
  __int64 v5; // rax
  void (__fastcall *v7)(_BYTE *, __int64, __int64); // rax

  a1[100] = a1[100] & 0xF8 | 1;
  v4 = !sub_32420F0((__int64)a1);
  v5 = *(_QWORD *)a1;
  if ( v4 )
  {
    v7 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(v5 + 8);
    if ( a2 > 31 )
    {
      v7(a1, 144, a3);
      return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 24LL))(a1, a2);
    }
    else
    {
      return ((__int64 (__fastcall *)(_BYTE *, _QWORD, __int64))v7)(a1, (unsigned __int8)(a2 + 80), a3);
    }
  }
  else
  {
    (*(void (__fastcall **)(_BYTE *, __int64, __int64))(v5 + 8))(a1, 144, a3);
    return (**(__int64 (__fastcall ***)(_BYTE *, _QWORD))a1)(a1, (unsigned int)a2);
  }
}
