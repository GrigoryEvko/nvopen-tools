// Function: sub_E0FA10
// Address: 0xe0fa10
//
__int64 __fastcall sub_E0FA10(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r12
  __int64 result; // rax

  v2 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v2 + 32LL))(v2);
  result = v2[9] & 0xC0;
  if ( (v2[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 40LL))(v2, a2);
  return result;
}
