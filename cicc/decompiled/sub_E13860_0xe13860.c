// Function: sub_E13860
// Address: 0xe13860
//
__int64 __fastcall sub_E13860(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  _BYTE *v5; // r13
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    goto LABEL_4;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 32LL))(v3);
  v4 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v4 + 9) & 0xC0) == 0x80 )
  {
    if ( (**(unsigned __int8 (__fastcall ***)(__int64, __int64 *))v4)(v4, a2) )
      goto LABEL_4;
  }
  else if ( (*(_BYTE *)(v4 + 9) & 0xC0) == 0 )
  {
    goto LABEL_4;
  }
  sub_E12F20(a2, 1u, " ");
LABEL_4:
  v5 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v5 + 32LL))(v5, a2);
  result = v5[9] & 0xC0;
  if ( (v5[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v5 + 40LL))(v5, a2);
  return result;
}
