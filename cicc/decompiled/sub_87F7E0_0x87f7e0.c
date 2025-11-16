// Function: sub_87F7E0
// Address: 0x87f7e0
//
_QWORD *__fastcall sub_87F7E0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  char *v5; // rax
  _QWORD *result; // rax

  v4 = sub_877070(a1, a2, a3, a4);
  v5 = (char *)sub_7279A0(10);
  strcpy(v5, "<unnamed>");
  *(_BYTE *)(v4 + 73) |= 1u;
  *(_QWORD *)(v4 + 8) = v5;
  *(_QWORD *)(v4 + 16) = 9;
  result = sub_87EBB0(a1, v4, a2);
  *((_DWORD *)result + 10) = *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
  return result;
}
