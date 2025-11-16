// Function: sub_1F6BF40
// Address: 0x1f6bf40
//
__int64 __fastcall sub_1F6BF40(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r13
  char v5; // r14
  __int64 v6; // rax

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE *)(a1 + 25);
  v6 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
  return sub_1F40B60(v4, a2, a3, v6, v5);
}
