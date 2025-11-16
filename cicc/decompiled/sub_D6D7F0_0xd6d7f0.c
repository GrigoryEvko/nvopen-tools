// Function: sub_D6D7F0
// Address: 0xd6d7f0
//
unsigned __int64 __fastcall sub_D6D7F0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // r13
  unsigned int v5; // ebx
  int v6; // ecx

  result = sub_D68B40(*a1, a3);
  if ( result )
  {
    v4 = result;
    v5 = 0;
    v6 = *(_DWORD *)(result + 4) & 0x7FFFFFF;
    if ( v6 )
    {
      do
      {
        if ( a2 == *(_QWORD *)(*(_QWORD *)(v4 - 8) + 32LL * *(unsigned int *)(v4 + 76) + 8LL * v5) )
        {
          sub_D68A80(v4, v5);
          v6 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
        }
        else
        {
          ++v5;
        }
      }
      while ( v6 != v5 );
    }
    return sub_D6D630((__int64)a1, v4);
  }
  return result;
}
