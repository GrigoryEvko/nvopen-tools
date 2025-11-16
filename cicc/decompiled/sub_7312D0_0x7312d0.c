// Function: sub_7312D0
// Address: 0x7312d0
//
_QWORD *__fastcall sub_7312D0(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rax

  if ( (*(_BYTE *)(a1 + 207) & 0x30) == 0x10 )
    sub_8B1A30(a1, dword_4F07508);
  v1 = sub_726700(20);
  v2 = sub_72D2E0(*(_QWORD **)(a1 + 152));
  v1[7] = a1;
  *v1 = v2;
  return v1;
}
