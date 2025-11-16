// Function: sub_8E5790
// Address: 0x8e5790
//
__int64 __fastcall sub_8E5790(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rcx
  unsigned __int64 v5; // r8

  while ( 1 )
  {
    result = *a1;
    if ( !(_BYTE)result )
      break;
    if ( !*(_QWORD *)(a2 + 32) )
    {
      v4 = *(_QWORD *)(a2 + 8);
      v2 = v4 + 1;
      if ( !*(_DWORD *)(a2 + 28) )
      {
        v5 = *(_QWORD *)(a2 + 16);
        if ( v5 <= v2 )
        {
          *(_DWORD *)(a2 + 28) = 1;
          if ( v5 )
          {
            *(_BYTE *)(*(_QWORD *)a2 + v5 - 1) = 0;
            v2 = *(_QWORD *)(a2 + 8) + 1LL;
          }
        }
        else
        {
          *(_BYTE *)(*(_QWORD *)a2 + v4) = result;
          v2 = *(_QWORD *)(a2 + 8) + 1LL;
        }
      }
      *(_QWORD *)(a2 + 8) = v2;
    }
    ++a1;
  }
  return result;
}
