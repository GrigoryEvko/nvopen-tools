// Function: sub_F99F90
// Address: 0xf99f90
//
bool __fastcall sub_F99F90(__int64 a1, __int64 *a2)
{
  unsigned int v2; // ebx
  __int64 v3; // r14
  int v4; // edx
  int v5; // eax
  __int64 v6; // r12
  unsigned int v7; // r13d
  unsigned int v8; // r15d
  bool result; // al
  __int64 v10; // r8
  int v11; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_QWORD *)a1;
  v11 = *(_DWORD *)(a1 + 24);
  v4 = v11 - v2;
  v5 = (int)(v11 - v2) >> 2;
  if ( v5 > 0 )
  {
    v6 = *a2;
    v7 = v2 + 4 * v5;
    while ( v6 != sub_B46EC0(v3, v2) )
    {
      v8 = v2 + 1;
      if ( v6 == sub_B46EC0(v3, v2 + 1) )
        return v11 != v8;
      v8 = v2 + 2;
      if ( v6 == sub_B46EC0(v3, v2 + 2) )
        return v11 != v8;
      v8 = v2 + 3;
      if ( v6 == sub_B46EC0(v3, v2 + 3) )
        return v11 != v8;
      v2 += 4;
      if ( v2 == v7 )
      {
        v4 = v11 - v2;
        goto LABEL_12;
      }
    }
    return v11 != v2;
  }
LABEL_12:
  if ( v4 != 2 )
  {
    if ( v4 != 3 )
    {
      result = 0;
      if ( v4 != 1 )
        return result;
      goto LABEL_15;
    }
    if ( *a2 == sub_B46EC0(v3, v2) )
      return v11 != v2;
    ++v2;
  }
  if ( *a2 == sub_B46EC0(v3, v2) )
    return v11 != v2;
  ++v2;
LABEL_15:
  v10 = sub_B46EC0(v3, v2);
  result = 0;
  if ( *a2 == v10 )
    return v11 != v2;
  return result;
}
