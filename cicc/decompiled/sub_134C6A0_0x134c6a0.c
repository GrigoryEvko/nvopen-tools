// Function: sub_134C6A0
// Address: 0x134c6a0
//
__int64 __fastcall sub_134C6A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax

  *(_BYTE *)(a2 + 160) = 1;
  sub_1636A40(a2, &unk_4F97BAC);
  sub_1636A40(a2, &unk_4F9B6E8);
  v2 = *(unsigned int *)(a2 + 152);
  if ( (unsigned int)v2 >= *(_DWORD *)(a2 + 156) )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    v2 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v2) = &unk_4F9B614;
  v3 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = v3;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)v3 )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    v3 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v3) = &unk_4F9D4AC;
  v4 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = v4;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)v4 )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    v4 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v4) = &unk_4F99BD0;
  v5 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = v5;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)v5 )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    v5 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v5) = &unk_4F98E5C;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = v6;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)v6 )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    v6 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v6) = &unk_4F9B60C;
  v7 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = v7;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)v7 )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    v7 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v7) = &unk_4F98A7C;
  result = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = result;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)result )
  {
    sub_16CD150(a2 + 144, a2 + 160, 0, 8);
    result = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * result) = &unk_4F98A84;
  ++*(_DWORD *)(a2 + 152);
  return result;
}
