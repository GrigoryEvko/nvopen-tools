// Function: sub_37B9A70
// Address: 0x37b9a70
//
__int64 __fastcall sub_37B9A70(__int64 a1, int *a2)
{
  __int64 v2; // r8
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rcx
  unsigned int v6; // esi

  v2 = *((_QWORD *)a2 + 2);
  v3 = *((_QWORD *)a2 + 1);
  result = 1;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *a2;
  if ( *(_DWORD *)a1 >= v6 )
  {
    result = 0;
    if ( *(_DWORD *)a1 == v6 )
    {
      result = 1;
      if ( v5 >= v3 )
      {
        LOBYTE(result) = *(_QWORD *)(a1 + 16) < v2;
        LOBYTE(v3) = v5 == v3;
        return (unsigned int)v3 & (unsigned int)result;
      }
    }
  }
  return result;
}
