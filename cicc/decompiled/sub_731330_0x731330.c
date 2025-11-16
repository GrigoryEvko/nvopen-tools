// Function: sub_731330
// Address: 0x731330
//
_QWORD *__fastcall sub_731330(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rax

  v1 = sub_726700(20);
  v2 = sub_72D2E0(*(_QWORD **)(a1 + 152));
  v1[7] = a1;
  *v1 = v2;
  *(_BYTE *)(a1 + 192) |= 1u;
  return v1;
}
