// Function: sub_1E47510
// Address: 0x1e47510
//
__int64 __fastcall sub_1E47510(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r8d
  __int64 v4; // rdi
  unsigned int v5; // edx
  __int64 v6; // rcx

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v3 = result - 1;
    v4 = *(_QWORD *)(a1 + 8);
    v5 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = *(_QWORD *)(v4 + 8LL * v5);
    result = 1;
    if ( v6 != a2 )
    {
      while ( 1 )
      {
        if ( v6 == -8 )
          return 0;
        v5 = v3 & (result + v5);
        v6 = *(_QWORD *)(v4 + 8LL * v5);
        if ( a2 == v6 )
          break;
        LODWORD(result) = result + 1;
      }
      return 1;
    }
  }
  return result;
}
