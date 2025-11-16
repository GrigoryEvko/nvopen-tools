// Function: sub_D6D880
// Address: 0xd6d880
//
unsigned __int64 __fastcall sub_D6D880(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // r13
  char v5; // r15
  unsigned int v6; // ebx
  int v7; // ecx

  result = sub_D68B40(*a1, a3);
  if ( result )
  {
    v4 = result;
    v5 = 0;
    v6 = 0;
    v7 = *(_DWORD *)(result + 4) & 0x7FFFFFF;
    if ( v7 )
    {
      do
      {
        if ( *(_QWORD *)(*(_QWORD *)(v4 - 8) + 32LL * *(unsigned int *)(v4 + 76) + 8LL * v6) == a2 )
        {
          if ( v5 )
          {
            sub_D68A80(v4, v6);
            v7 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
          }
          else
          {
            ++v6;
            v5 = 1;
          }
        }
        else
        {
          ++v6;
        }
      }
      while ( v7 != v6 );
    }
    return sub_D6D630((__int64)a1, v4);
  }
  return result;
}
