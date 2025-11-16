// Function: sub_880E20
// Address: 0x880e20
//
_QWORD *__fastcall sub_880E20(__int64 *a1)
{
  __int64 v1; // rax
  _QWORD *v3; // r12
  __int64 v4; // rax

  v1 = a1[12];
  if ( v1 )
    return *(_QWORD **)(v1 + 32);
  v3 = sub_87EBB0(0x14u, *a1, a1 + 6);
  *((_BYTE *)v3 + 81) |= 0x10u;
  v3[8] = a1[8];
  *(_QWORD *)(v3[11] + 176LL) = a1[11];
  v4 = sub_880C60();
  *(_QWORD *)(v4 + 32) = v3;
  *(_QWORD *)(v4 + 24) = a1;
  a1[12] = v4;
  return v3;
}
