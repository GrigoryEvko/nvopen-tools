// Function: sub_1A7A090
// Address: 0x1a7a090
//
__int64 __fastcall sub_1A7A090(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 result; // rax

  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4F99CB0);
  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
