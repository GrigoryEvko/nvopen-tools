// Function: sub_1CCFFA0
// Address: 0x1ccffa0
//
__int64 __fastcall sub_1CCFFA0(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 result; // rax

  sub_1636A40(a2, (__int64)&unk_4F9920C);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  sub_1636A40(a2, (__int64)&unk_4FB9E2C);
  sub_1636A40(a2, (__int64)&unk_4FB9E34);
  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4FB9E34;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
