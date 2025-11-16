// Function: sub_17890F0
// Address: 0x17890f0
//
__int64 __fastcall sub_17890F0(_QWORD *a1, _DWORD *a2)
{
  __int64 result; // rax
  unsigned int v3; // ecx
  __int64 *v4; // r13
  unsigned int v5; // eax
  __int64 *v6; // rbx
  unsigned int v7; // ebx

  result = 0xFFFFFFFFLL;
  if ( *(_DWORD *)a1 >= *a2 )
  {
    result = 1;
    if ( *(_DWORD *)a1 <= *a2 )
    {
      result = 0xFFFFFFFFLL;
      v3 = a2[1];
      if ( *((_DWORD *)a1 + 1) >= v3 )
      {
        result = 1;
        if ( *((_DWORD *)a1 + 1) <= v3 )
        {
          v4 = (__int64 *)a1[1];
          v5 = sub_1643030(*v4);
          v6 = (__int64 *)*((_QWORD *)a2 + 1);
          if ( v5 < (unsigned int)sub_1643030(*v6) )
          {
            return 0xFFFFFFFFLL;
          }
          else
          {
            v7 = sub_1643030(*v6);
            return v7 < (unsigned int)sub_1643030(*v4);
          }
        }
      }
    }
  }
  return result;
}
