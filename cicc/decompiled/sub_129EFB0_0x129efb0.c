// Function: sub_129EFB0
// Address: 0x129efb0
//
_QWORD *__fastcall sub_129EFB0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax

  v3 = *(_QWORD *)(a3 + 128);
  if ( v3 && (*(_BYTE *)(v3 + 89) & 8) != 0 )
    a3 = *(_QWORD *)(a3 + 128);
  sub_129EC20(a1, a2, a3, *(_QWORD *)(a3 + 240));
  return a1;
}
