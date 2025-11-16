// Function: sub_B7E5C0
// Address: 0xb7e5c0
//
__int64 __fastcall sub_B7E5C0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned int v3; // ebx
  __int64 v4; // rax

  result = *(unsigned int *)(a1 + 608);
  if ( (_DWORD)result )
  {
    v3 = 0;
    do
    {
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v3);
      if ( !v4 )
        BUG();
      ++v3;
      result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)(v4 - 176) + 136LL))(v4 - 176, a2);
    }
    while ( *(_DWORD *)(a1 + 608) > v3 );
  }
  return result;
}
