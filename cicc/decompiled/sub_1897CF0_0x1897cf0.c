// Function: sub_1897CF0
// Address: 0x1897cf0
//
bool __fastcall sub_1897CF0(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // ebx
  __int64 v3; // r13
  int v4; // edx
  int v5; // eax
  unsigned int v6; // r14d
  unsigned int v7; // r15d
  int v9; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_QWORD *)a1;
  v9 = *(_DWORD *)(a1 + 24);
  v4 = v9 - v2;
  v5 = (int)(v9 - v2) >> 2;
  if ( v5 > 0 )
  {
    v6 = v2 + 4 * v5;
    while ( *a2 != sub_15F4DF0(v3, v2) )
    {
      v7 = v2 + 1;
      if ( *a2 == sub_15F4DF0(v3, v2 + 1) )
        return v9 != v7;
      v7 = v2 + 2;
      if ( *a2 == sub_15F4DF0(v3, v2 + 2) )
        return v9 != v7;
      v7 = v2 + 3;
      if ( *a2 == sub_15F4DF0(v3, v2 + 3) )
        return v9 != v7;
      v2 += 4;
      if ( v2 == v6 )
      {
        v4 = v9 - v2;
        goto LABEL_11;
      }
    }
    return v9 != v2;
  }
LABEL_11:
  if ( v4 == 2 )
    goto LABEL_17;
  if ( v4 == 3 )
  {
    if ( *a2 == sub_15F4DF0(v3, v2) )
      return v9 != v2;
    ++v2;
LABEL_17:
    if ( *a2 != sub_15F4DF0(v3, v2) )
    {
      ++v2;
      goto LABEL_19;
    }
    return v9 != v2;
  }
  if ( v4 != 1 )
    return 0;
LABEL_19:
  if ( *a2 == sub_15F4DF0(v3, v2) )
    return v9 != v2;
  return 0;
}
