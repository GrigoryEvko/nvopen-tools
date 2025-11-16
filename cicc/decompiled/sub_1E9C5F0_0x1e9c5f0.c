// Function: sub_1E9C5F0
// Address: 0x1e9c5f0
//
__int64 __fastcall sub_1E9C5F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r8d
  int v4; // r9d
  __int64 result; // rax
  int v6; // r8d
  int v7; // r9d

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  sub_1E11F70(a1, a2);
  sub_1636A40(a2, (__int64)&unk_4FC6A0C);
  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  if ( byte_4FC89E0 )
  {
    sub_1636A40(a2, (__int64)&unk_4FC62EC);
    result = *(unsigned int *)(a2 + 120);
    if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
    {
      sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
      result = *(unsigned int *)(a2 + 120);
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4FC62EC;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}
