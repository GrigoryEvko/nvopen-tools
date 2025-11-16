// Function: sub_18A9FE0
// Address: 0x18a9fe0
//
__int64 __fastcall sub_18A9FE0(_QWORD *a1, __int64 a2, unsigned int *a3)
{
  unsigned int v4; // r14d
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // eax
  unsigned int v12; // eax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_18A9F10((__int64)a1, a3);
    v10 = a1[4];
    v11 = *(_DWORD *)(v10 + 32);
    if ( *a3 <= v11 && (*a3 != v11 || a3[1] <= *(_DWORD *)(v10 + 36)) )
      return sub_18A9F10((__int64)a1, a3);
    return 0;
  }
  v4 = *a3;
  if ( *(_DWORD *)(a2 + 32) > *a3 )
    goto LABEL_3;
  if ( *(_DWORD *)(a2 + 32) != v4 )
  {
    if ( *(_DWORD *)(a2 + 32) < v4 )
      goto LABEL_10;
    return a2;
  }
  v12 = a3[1];
  if ( *(_DWORD *)(a2 + 36) > v12 )
  {
LABEL_3:
    result = a2;
    if ( a1[3] == a2 )
      return result;
    v6 = sub_220EF80(a2);
    v7 = v6;
    if ( v4 > *(_DWORD *)(v6 + 32) || v4 == *(_DWORD *)(v6 + 32) && *(_DWORD *)(v6 + 36) < a3[1] )
    {
      result = 0;
      if ( *(_QWORD *)(v7 + 24) )
        return a2;
      return result;
    }
    return sub_18A9F10((__int64)a1, a3);
  }
  if ( *(_DWORD *)(a2 + 36) >= v12 )
    return a2;
LABEL_10:
  if ( a1[4] == a2 )
    return 0;
  v8 = sub_220EEE0(a2);
  v9 = v8;
  if ( v4 >= *(_DWORD *)(v8 + 32) && (v4 != *(_DWORD *)(v8 + 32) || a3[1] >= *(_DWORD *)(v8 + 36)) )
    return sub_18A9F10((__int64)a1, a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v9;
  return result;
}
