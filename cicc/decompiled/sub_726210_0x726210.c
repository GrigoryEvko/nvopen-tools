// Function: sub_726210
// Address: 0x726210
//
__int64 __fastcall sub_726210(__int64 a1)
{
  _QWORD *v1; // rax

  v1 = sub_7247C0(48);
  *(_QWORD *)(a1 + 256) = v1;
  *v1 = 0;
  v1[1] = 0;
  v1[2] = 0;
  v1[3] = 0;
  *((_DWORD *)v1 + 8) = 0;
  v1[5] = 0;
  return *(_QWORD *)(a1 + 256);
}
