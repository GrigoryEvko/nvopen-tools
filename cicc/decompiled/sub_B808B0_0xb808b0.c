// Function: sub_B808B0
// Address: 0xb808b0
//
__int64 __fastcall sub_B808B0(__int64 a1)
{
  unsigned int v2; // r13d
  __int64 v3; // r12
  __int64 result; // rax
  unsigned int v5; // ebx
  __int64 v6; // rdx
  __int64 v7; // rdi

  if ( *(_BYTE *)(a1 + 1288) )
  {
    if ( *(_DWORD *)(a1 + 608) )
    {
      v2 = 0;
      do
      {
        v3 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v2);
        if ( !v3 )
          BUG();
        result = *(unsigned int *)(v3 + 24);
        if ( (_DWORD)result )
        {
          v5 = 0;
          do
          {
            v6 = v5++;
            v7 = *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * v6);
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 96LL))(v7);
          }
          while ( *(_DWORD *)(v3 + 24) > v5 );
        }
        ++v2;
      }
      while ( *(_DWORD *)(a1 + 608) > v2 );
    }
    *(_BYTE *)(a1 + 1288) = 0;
  }
  return result;
}
