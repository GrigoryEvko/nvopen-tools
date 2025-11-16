// Function: sub_18EE1B0
// Address: 0x18ee1b0
//
__int64 __fastcall sub_18EE1B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 result; // rax

  v2 = a2 + 112;
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F99130);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F98E5C;
  result = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = result;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)result )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F9E06C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
