// Function: sub_32432C0
// Address: 0x32432c0
//
__int64 __fastcall sub_32432C0(_BYTE *a1, __int64 a2)
{
  a1[100] = a1[100] & 0xF8 | 3;
  (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 17, 0);
  return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 16LL))(a1, a2);
}
