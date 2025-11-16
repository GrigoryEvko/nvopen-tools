// Function: sub_197D0A0
// Address: 0x197d0a0
//
__int64 __fastcall sub_197D0A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rax
  __int64 result; // rax

  v2 = a2 + 112;
  sub_1636A40(a2, (__int64)&unk_4FB66D8);
  sub_1636A40(a2, (__int64)&unk_4F9920C);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F9920C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_5051F8C);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  v8 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4F9E06C;
  result = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = result;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)result )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
