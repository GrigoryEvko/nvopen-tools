// Function: sub_1E77E90
// Address: 0x1e77e90
//
__int64 __fastcall sub_1E77E90(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 result; // rax

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  sub_1E11F70(a1, a2);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4FC62EC);
  sub_1636A40(a2, (__int64)&unk_4FC71EC);
  sub_1636A40(a2, (__int64)&unk_4FC6A0C);
  sub_1636A40(a2, (__int64)&unk_4FC5828);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4FC62EC;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v6;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v6 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4FC71EC;
  result = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = result;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)result )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  if ( byte_4FC8100 )
    return sub_1636A40(a2, (__int64)&unk_4FC453D);
  return result;
}
