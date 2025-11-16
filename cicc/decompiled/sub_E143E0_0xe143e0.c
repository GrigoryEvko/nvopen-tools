// Function: sub_E143E0
// Address: 0xe143e0
//
__int64 __fastcall sub_E143E0(__int64 a1, __int64 *a2)
{
  _BYTE *v2; // r13
  _BYTE *v3; // r13
  __int64 result; // rax
  _BYTE *v5; // r13

  if ( *(_BYTE *)(a1 + 32) )
  {
    sub_E14360((__int64)a2, 91);
    v2 = *(_BYTE **)(a1 + 16);
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v2 + 32LL))(v2, a2);
    if ( (v2[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v2 + 40LL))(v2, a2);
    sub_E14360((__int64)a2, 93);
  }
  else
  {
    sub_E14360((__int64)a2, 46);
    v5 = *(_BYTE **)(a1 + 16);
    (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v5 + 32LL))(v5, a2);
    if ( (v5[9] & 0xC0) != 0x40 )
    {
      (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v5 + 40LL))(v5, a2);
      v3 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int8)(v3[8] - 81) <= 1u )
        goto LABEL_6;
      goto LABEL_10;
    }
  }
  v3 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int8)(v3[8] - 81) <= 1u )
    goto LABEL_6;
LABEL_10:
  sub_E12F20(a2, 3u, " = ");
  v3 = *(_BYTE **)(a1 + 24);
LABEL_6:
  (*(void (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v3 + 32LL))(v3, a2);
  result = v3[9] & 0xC0;
  if ( (v3[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, __int64 *))(*(_QWORD *)v3 + 40LL))(v3, a2);
  return result;
}
