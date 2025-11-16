// Function: sub_B823B0
// Address: 0xb823b0
//
__int64 __fastcall sub_B823B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v5; // edx
  __int64 v6; // rdi

  v6 = *(_QWORD *)(a1 + 8);
  result = *(unsigned int *)(v6 + 608);
  *(_QWORD *)(v6 + 1288) = a2;
  *(_QWORD *)(v6 + 1296) = a3;
  if ( (_DWORD)result )
  {
    v5 = 0;
    do
    {
      result = *(_QWORD *)(*(_QWORD *)(v6 + 600) + 8LL * v5);
      if ( !result )
      {
        MEMORY[0x268] = a2;
        BUG();
      }
      *(_QWORD *)(result + 440) = a2;
      ++v5;
      *(_QWORD *)(result + 448) = a3;
    }
    while ( *(_DWORD *)(v6 + 608) > v5 );
  }
  return result;
}
