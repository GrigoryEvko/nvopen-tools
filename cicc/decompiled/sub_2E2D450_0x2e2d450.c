// Function: sub_2E2D450
// Address: 0x2e2d450
//
__int64 __fastcall sub_2E2D450(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rax
  bool v3; // zf
  __int64 result; // rax
  int v5; // eax

  v2 = a2[1];
  v3 = a1[1] == v2;
  if ( a1[1] < v2 )
    return 0xFFFFFFFFLL;
  result = 1;
  if ( v3 )
  {
    v5 = *((_DWORD *)a2 + 4);
    if ( *((_DWORD *)a1 + 4) < v5 )
      return 0xFFFFFFFFLL;
    if ( *((_DWORD *)a1 + 4) == v5 )
    {
      if ( *((_DWORD *)a1 + 5) < *((_DWORD *)a2 + 5) )
        return 0xFFFFFFFFLL;
      if ( *((_DWORD *)a2 + 5) >= *((_DWORD *)a1 + 5) )
        return 0;
    }
    else if ( *((_DWORD *)a1 + 4) <= v5 )
    {
      return 0;
    }
    return 1;
  }
  return result;
}
