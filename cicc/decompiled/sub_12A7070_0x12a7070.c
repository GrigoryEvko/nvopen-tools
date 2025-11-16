// Function: sub_12A7070
// Address: 0x12a7070
//
__int64 __fastcall sub_12A7070(__int64 a1, unsigned __int64 *a2, unsigned __int64 *a3)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r12
  __int64 result; // rax
  _QWORD v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 16) + 16LL);
  if ( *(_BYTE *)(v5 + 24) != 2
    || (v7 = *(_QWORD *)(v5 + 56), *(_BYTE *)(v7 + 173) != 1)
    || *(_BYTE *)(v6 + 24) != 2
    || *(_BYTE *)(*(_QWORD *)(v6 + 56) + 173LL) != 1 )
  {
    sub_127B550("unexpected non-int-const operand in p2r/r2p", (_DWORD *)(a1 + 36), 1);
  }
  v8 = sub_620FD0(v7, v11);
  v9 = sub_620FD0(*(_QWORD *)(v6 + 56), (_DWORD *)v11 + 1);
  result = (unsigned int)(HIDWORD(v11[0]) | LODWORD(v11[0]));
  if ( v11[0] )
    sub_127B550("unexpected constant overflow in p2r/r2p operand", (_DWORD *)(a1 + 36), 1);
  if ( v8 > 3 )
    sub_127B550("expected byte-idx operand to be in 0-3", (_DWORD *)(a1 + 36), 1);
  if ( v9 > 0x7F )
    sub_127B550("expected mask operand to be 0-127", (_DWORD *)(a1 + 36), 1);
  *a2 = v8;
  *a3 = v9;
  return result;
}
