// Function: sub_186F010
// Address: 0x186f010
//
__int64 __fastcall sub_186F010(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 result; // rax

  sub_1636A10(a2, a2);
  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F98A8D;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
