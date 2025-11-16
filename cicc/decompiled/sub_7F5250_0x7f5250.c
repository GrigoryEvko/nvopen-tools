// Function: sub_7F5250
// Address: 0x7f5250
//
_QWORD *__fastcall sub_7F5250(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r12

  v4 = sub_7259C0(7);
  v4[20] = a1;
  v5 = v4;
  *(_BYTE *)(v4[21] + 16LL) = (2 * (unk_4F06968 == 0)) | *(_BYTE *)(v4[21] + 16LL) & 0xFD;
  if ( a2 )
  {
    *(_QWORD *)v4[21] = sub_724EF0(a2);
    if ( a3 )
      **(_QWORD **)v5[21] = sub_724EF0(a3);
  }
  return v5;
}
