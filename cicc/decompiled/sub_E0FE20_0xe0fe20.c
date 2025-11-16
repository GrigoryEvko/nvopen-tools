// Function: sub_E0FE20
// Address: 0xe0fe20
//
__int64 __fastcall sub_E0FE20(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r13
  _BYTE *v3; // r13
  __int64 result; // rax

  v2 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v2 + 32LL))(v2);
  if ( (v2[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 40LL))(v2, a2);
  v3 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 32LL))(v3, a2);
  result = v3[9] & 0xC0;
  if ( (v3[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
  return result;
}
