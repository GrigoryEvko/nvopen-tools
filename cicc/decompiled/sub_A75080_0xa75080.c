// Function: sub_A75080
// Address: 0xa75080
//
__int64 __fastcall sub_A75080(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  _QWORD *v6; // rcx

  v2 = *(unsigned int *)(a1 + 16);
  result = 0;
  if ( v2 == *(_DWORD *)(a2 + 16) )
  {
    v4 = *(_QWORD **)(a1 + 8);
    v5 = *(_QWORD **)(a2 + 8);
    v6 = &v4[v2];
    if ( v4 == v6 )
    {
      return 1;
    }
    else
    {
      while ( *v5 == *v4 )
      {
        ++v4;
        ++v5;
        if ( v6 == v4 )
          return 1;
      }
      return 0;
    }
  }
  return result;
}
