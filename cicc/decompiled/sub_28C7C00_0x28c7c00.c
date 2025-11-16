// Function: sub_28C7C00
// Address: 0x28c7c00
//
__int64 __fastcall sub_28C7C00(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax
  int v3; // eax
  int v4; // eax
  int v5; // edx
  __int64 v6; // rcx
  __int64 v7; // rax

  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  result = 1;
  if ( *a1 == *a2 )
  {
    v3 = a2[1];
    if ( a1[1] >= v3 )
    {
      if ( a1[1] != v3 )
        return a1[1] > v3;
      v4 = a1[2];
      v5 = a2[2];
      if ( v4 >= v5 )
      {
        if ( v4 != v5
          || (v6 = *((_QWORD *)a2 + 2), *((_QWORD *)a1 + 2) >= v6)
          && (*((_QWORD *)a1 + 2) != v6 || *((_QWORD *)a1 + 3) >= *((_QWORD *)a2 + 3)) )
        {
          if ( v5 < v4 )
            return 1;
          if ( v5 == v4 )
          {
            v7 = *((_QWORD *)a1 + 2);
            if ( *((_QWORD *)a2 + 2) < v7 || *((_QWORD *)a2 + 2) == v7 && *((_QWORD *)a2 + 3) < *((_QWORD *)a1 + 3) )
              return 1;
          }
          return 0;
        }
      }
    }
    return 0xFFFFFFFFLL;
  }
  return result;
}
