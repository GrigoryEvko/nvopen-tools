// Function: sub_12394C0
// Address: 0x12394c0
//
__int64 __fastcall sub_12394C0(_QWORD *a1, __int64 a2, unsigned int *a3)
{
  unsigned int v4; // r14d
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v8; // rax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] || *a3 <= *(_DWORD *)(a1[4] + 32LL) )
      return sub_1239420((__int64)a1, a3);
  }
  else
  {
    v4 = *a3;
    if ( *(_DWORD *)(a2 + 32) > *a3 )
    {
      v5 = a1[3];
      if ( v5 == a2 )
        return v5;
      v6 = sub_220EF80(a2);
      if ( *(_DWORD *)(v6 + 32) < v4 )
      {
        v5 = 0;
        if ( *(_QWORD *)(v6 + 24) )
          return a2;
        return v5;
      }
      return sub_1239420((__int64)a1, a3);
    }
    if ( *(_DWORD *)(a2 + 32) >= v4 )
      return a2;
    if ( a1[4] != a2 )
    {
      v8 = sub_220EEE0(a2);
      if ( v4 < *(_DWORD *)(v8 + 32) )
      {
        v5 = 0;
        if ( *(_QWORD *)(a2 + 24) )
          return v8;
        return v5;
      }
      return sub_1239420((__int64)a1, a3);
    }
  }
  return 0;
}
