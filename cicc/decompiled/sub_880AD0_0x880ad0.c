// Function: sub_880AD0
// Address: 0x880ad0
//
_QWORD *__fastcall sub_880AD0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // r12
  char v3; // al

  v1 = (_QWORD *)sub_823970(136);
  *v1 = 0;
  v2 = v1;
  v1[1] = a1;
  sub_879020((__int64)(v1 + 2), 1);
  *((_WORD *)v2 + 28) &= 0xF000u;
  v3 = *(_BYTE *)(a1 + 80);
  if ( v3 == 3 )
  {
    v2[8] = *(_QWORD *)(a1 + 88);
  }
  else
  {
    v2[8] = *(_QWORD *)(a1 + 88);
    if ( v3 == 2 )
      *((_BYTE *)v2 + 72) &= ~1u;
  }
  v2[11] = 0;
  v2[10] = 0;
  sub_879020((__int64)(v2 + 12), 1);
  *((_DWORD *)v2 + 15) = 0;
  return v2;
}
