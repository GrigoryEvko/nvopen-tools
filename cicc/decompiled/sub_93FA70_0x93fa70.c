// Function: sub_93FA70
// Address: 0x93fa70
//
_QWORD *__fastcall sub_93FA70(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax

  v3 = *(_QWORD *)(a3 + 128);
  if ( v3 && (*(_BYTE *)(v3 + 89) & 8) != 0 )
    a3 = *(_QWORD *)(a3 + 128);
  sub_93F6E0(a1, a2, a3, *(_QWORD *)(a3 + 240));
  return a1;
}
