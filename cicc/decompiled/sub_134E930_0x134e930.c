// Function: sub_134E930
// Address: 0x134e930
//
__int64 __fastcall sub_134E930(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 result; // rax

  v1 = a1 + 144;
  sub_1636A40(a1, &unk_4F9B6E8);
  v2 = *(unsigned int *)(a1 + 152);
  if ( (unsigned int)v2 >= *(_DWORD *)(a1 + 156) )
  {
    sub_16CD150(a1 + 144, a1 + 160, 0, 8);
    v2 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v2) = &unk_4F9B614;
  v3 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
  *(_DWORD *)(a1 + 152) = v3;
  if ( *(_DWORD *)(a1 + 156) <= (unsigned int)v3 )
  {
    sub_16CD150(v1, a1 + 160, 0, 8);
    v3 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v3) = &unk_4F9D4AC;
  v4 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
  *(_DWORD *)(a1 + 152) = v4;
  if ( *(_DWORD *)(a1 + 156) <= (unsigned int)v4 )
  {
    sub_16CD150(v1, a1 + 160, 0, 8);
    v4 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v4) = &unk_4F99BD0;
  v5 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
  *(_DWORD *)(a1 + 152) = v5;
  if ( *(_DWORD *)(a1 + 156) <= (unsigned int)v5 )
  {
    sub_16CD150(v1, a1 + 160, 0, 8);
    v5 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v5) = &unk_4F98E5C;
  v6 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
  *(_DWORD *)(a1 + 152) = v6;
  if ( *(_DWORD *)(a1 + 156) <= (unsigned int)v6 )
  {
    sub_16CD150(v1, a1 + 160, 0, 8);
    v6 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v6) = &unk_4F98A7C;
  result = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
  *(_DWORD *)(a1 + 152) = result;
  if ( *(_DWORD *)(a1 + 156) <= (unsigned int)result )
  {
    sub_16CD150(v1, a1 + 160, 0, 8);
    result = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * result) = &unk_4F98A84;
  ++*(_DWORD *)(a1 + 152);
  return result;
}
