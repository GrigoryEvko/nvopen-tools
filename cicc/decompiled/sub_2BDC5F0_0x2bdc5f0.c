// Function: sub_2BDC5F0
// Address: 0x2bdc5f0
//
bool __fastcall sub_2BDC5F0(__int64 a1, char *a2)
{
  char v2; // bl
  unsigned int v3; // r12d
  __int64 v4; // rax

  v2 = *(_BYTE *)(a1 + 8);
  v3 = *a2;
  v4 = sub_222F790(*(_QWORD **)a1, (__int64)a2);
  return v2 == (*(char (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 32LL))(v4, v3);
}
