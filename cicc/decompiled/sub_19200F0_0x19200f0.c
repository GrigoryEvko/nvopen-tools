// Function: sub_19200F0
// Address: 0x19200f0
//
__int64 __fastcall sub_19200F0(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 result; // rax

  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F99CCC);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4F99308);
  sub_1636A40(a2, (__int64)&unk_4F99768);
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4F9E06C;
  v5 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v5;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v5 )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F99768;
  result = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = result;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)result )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
