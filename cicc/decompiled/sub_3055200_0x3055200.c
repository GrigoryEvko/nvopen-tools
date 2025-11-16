// Function: sub_3055200
// Address: 0x3055200
//
__int64 __fastcall sub_3055200(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // eax
  unsigned int v12; // edx

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_3055070((__int64)a1, (unsigned __int64 *)a3);
    v10 = a1[4];
    if ( *(_QWORD *)(v10 + 32) >= *(_QWORD *)a3
      && (*(_QWORD *)(v10 + 32) != *(_QWORD *)a3 || *(_DWORD *)(v10 + 40) >= *(_DWORD *)(a3 + 8)) )
    {
      return sub_3055070((__int64)a1, (unsigned __int64 *)a3);
    }
    return 0;
  }
  v4 = *(_QWORD *)a3;
  if ( *(_QWORD *)a3 < *(_QWORD *)(a2 + 32) )
    goto LABEL_3;
  if ( *(_QWORD *)a3 == *(_QWORD *)(a2 + 32) )
  {
    v11 = *(_DWORD *)(a3 + 8);
    v12 = *(_DWORD *)(a2 + 40);
    if ( v11 < v12 )
    {
LABEL_3:
      if ( a1[3] == a2 )
        return a2;
      v5 = sub_220EF80(a2);
      v6 = v5;
      if ( v4 > *(_QWORD *)(v5 + 32) || v4 == *(_QWORD *)(v5 + 32) && *(_DWORD *)(v5 + 40) < *(_DWORD *)(a3 + 8) )
      {
        result = 0;
        if ( *(_QWORD *)(v6 + 24) )
          return a2;
        return result;
      }
      return sub_3055070((__int64)a1, (unsigned __int64 *)a3);
    }
  }
  else
  {
    if ( *(_QWORD *)a3 > *(_QWORD *)(a2 + 32) )
      goto LABEL_12;
    v11 = *(_DWORD *)(a3 + 8);
    v12 = *(_DWORD *)(a2 + 40);
  }
  if ( v12 >= v11 )
    return a2;
LABEL_12:
  if ( a1[4] == a2 )
    return 0;
  v8 = sub_220EEE0(a2);
  v9 = v8;
  if ( v4 >= *(_QWORD *)(v8 + 32) && (v4 != *(_QWORD *)(v8 + 32) || *(_DWORD *)(a3 + 8) >= *(_DWORD *)(v8 + 40)) )
    return sub_3055070((__int64)a1, (unsigned __int64 *)a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v9;
  return result;
}
