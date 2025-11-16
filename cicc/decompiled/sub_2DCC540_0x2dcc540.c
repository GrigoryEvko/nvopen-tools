// Function: sub_2DCC540
// Address: 0x2dcc540
//
__int64 __fastcall sub_2DCC540(_QWORD *a1, __int64 a2, unsigned int *a3)
{
  unsigned int v4; // r13d
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  unsigned int v12; // eax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_2DCB9E0((__int64)a1, a3);
    v11 = a1[4];
    v12 = *(_DWORD *)(v11 + 32);
    if ( *a3 <= v12 && (*a3 != v12 || (signed int)a3[1] <= *(_DWORD *)(v11 + 36)) )
      return sub_2DCB9E0((__int64)a1, a3);
    return 0;
  }
  v4 = *a3;
  v5 = *(_DWORD *)(a2 + 32);
  if ( v5 > *a3 || v5 == v4 && *(_DWORD *)(a2 + 36) > (signed int)a3[1] )
  {
    result = a2;
    if ( a1[3] == a2 )
      return result;
    v7 = sub_220EF80(a2);
    v8 = v7;
    if ( v4 > *(_DWORD *)(v7 + 32) || v4 == *(_DWORD *)(v7 + 32) && *(_DWORD *)(v7 + 36) < (signed int)a3[1] )
    {
      result = 0;
      if ( *(_QWORD *)(v8 + 24) )
        return a2;
      return result;
    }
    return sub_2DCB9E0((__int64)a1, a3);
  }
  if ( v5 >= v4 && *(_DWORD *)(a2 + 36) >= (signed int)a3[1] )
    return a2;
  if ( a1[4] == a2 )
    return 0;
  v9 = sub_220EEE0(a2);
  v10 = v9;
  if ( v4 >= *(_DWORD *)(v9 + 32) && (v4 != *(_DWORD *)(v9 + 32) || (signed int)a3[1] >= *(_DWORD *)(v9 + 36)) )
    return sub_2DCB9E0((__int64)a1, a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v10;
  return result;
}
