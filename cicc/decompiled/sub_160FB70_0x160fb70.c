// Function: sub_160FB70
// Address: 0x160fb70
//
__int64 __fastcall sub_160FB70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v5; // edx
  __int64 v6; // rdi

  v6 = *(_QWORD *)(a1 + 16);
  result = *(unsigned int *)(v6 + 608);
  *(_QWORD *)(v6 + 1304) = a2;
  *(_QWORD *)(v6 + 1312) = a3;
  if ( (_DWORD)result )
  {
    v5 = 0;
    do
    {
      result = *(_QWORD *)(*(_QWORD *)(v6 + 600) + 8LL * v5);
      if ( !result )
      {
        MEMORY[0x270] = a2;
        BUG();
      }
      *(_QWORD *)(result + 464) = a2;
      ++v5;
      *(_QWORD *)(result + 472) = a3;
    }
    while ( *(_DWORD *)(v6 + 608) > v5 );
  }
  return result;
}
