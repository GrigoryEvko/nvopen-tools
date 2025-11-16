// Function: sub_1AED640
// Address: 0x1aed640
//
__int64 __fastcall sub_1AED640(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // ebx
  __int64 v3; // r15
  int v4; // edx
  int v5; // eax
  unsigned int v6; // r13d
  __int64 v8; // [rsp+0h] [rbp-40h]
  int v9; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_QWORD *)a1;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_DWORD *)(a1 + 24);
  v4 = v9 - v2;
  v5 = (int)(v9 - v2) >> 2;
  if ( v5 <= 0 )
  {
LABEL_11:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          return v8;
LABEL_19:
        if ( *a2 == sub_15F4DF0(v3, v2) )
          return v3;
        return v8;
      }
      if ( *a2 == sub_15F4DF0(v3, v2) )
        return v3;
      ++v2;
    }
    if ( *a2 == sub_15F4DF0(v3, v2) )
      return v3;
    ++v2;
    goto LABEL_19;
  }
  v6 = v2 + 4 * v5;
  while ( *a2 != sub_15F4DF0(v3, v2)
       && *a2 != sub_15F4DF0(v3, v2 + 1)
       && *a2 != sub_15F4DF0(v3, v2 + 2)
       && *a2 != sub_15F4DF0(v3, v2 + 3) )
  {
    v2 += 4;
    if ( v2 == v6 )
    {
      v4 = v9 - v2;
      goto LABEL_11;
    }
  }
  return v3;
}
