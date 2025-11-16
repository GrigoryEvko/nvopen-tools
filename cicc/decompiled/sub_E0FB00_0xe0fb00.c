// Function: sub_E0FB00
// Address: 0xe0fb00
//
__int64 __fastcall sub_E0FB00(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r13

  v2 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v2 + 32LL))(v2);
  if ( (v2[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 40LL))(v2, a2);
  return (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 24) + 40LL))(*(_QWORD *)(a1 + 24), a2);
}
