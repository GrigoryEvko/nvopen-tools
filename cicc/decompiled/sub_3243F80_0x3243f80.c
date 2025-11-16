// Function: sub_3243F80
// Address: 0x3243f80
//
__int64 __fastcall sub_3243F80(_BYTE *a1, unsigned int a2, __int64 a3)
{
  void (__fastcall *v4)(_BYTE *, _QWORD); // rax
  __int64 result; // rax

  (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 237, 0);
  v4 = *(void (__fastcall **)(_BYTE *, _QWORD))(*(_QWORD *)a1 + 24LL);
  if ( a2 == 4 )
  {
    v4(a1, 0);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 24LL))(a1, a3);
    result = a1[100] & 0xF8 | 2u;
    a1[100] = a1[100] & 0xF8 | 2;
  }
  else
  {
    v4(a1, a2);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 24LL))(a1, a3);
    result = a1[100] & 0xF8 | 3u;
    a1[100] = a1[100] & 0xF8 | 3;
  }
  return result;
}
