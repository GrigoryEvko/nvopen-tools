// Function: sub_3157250
// Address: 0x3157250
//
__int64 __fastcall sub_3157250(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r8
  _DWORD *v3; // rcx
  __int64 v4; // rdx
  int v5; // edx

  result = *(unsigned int *)(a1 + 48);
  if ( !(_DWORD)result )
  {
    v2 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v2 )
    {
      for ( result = 0; result != v2; ++result )
      {
        while ( 1 )
        {
          v3 = (_DWORD *)(*(_QWORD *)a1 + 4 * result);
          v4 = (unsigned int)*v3;
          if ( (_DWORD)v4 == (_DWORD)result )
            break;
          ++result;
          *v3 = *(_DWORD *)(*(_QWORD *)a1 + 4 * v4);
          if ( v2 == result )
            return result;
        }
        v5 = *(_DWORD *)(a1 + 48);
        *(_DWORD *)(a1 + 48) = v5 + 1;
        *v3 = v5;
      }
    }
  }
  return result;
}
