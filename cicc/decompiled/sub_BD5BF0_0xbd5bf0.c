// Function: sub_BD5BF0
// Address: 0xbd5bf0
//
__int64 __fastcall sub_BD5BF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rcx
  int v6; // esi
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx

  result = a1;
  if ( *(_BYTE *)a1 == 84 && *(_QWORD *)(a1 + 40) == a2 )
  {
    v4 = *(_QWORD *)(a1 - 8);
    v5 = 0x1FFFFFFFE0LL;
    v6 = *(_DWORD *)(result + 4) & 0x7FFFFFF;
    if ( v6 )
    {
      v7 = *(unsigned int *)(result + 72);
      v8 = 0;
      v9 = v4 + 32 * v7;
      do
      {
        if ( a3 == *(_QWORD *)(v9 + 8 * v8) )
        {
          v5 = 32 * v8;
          return *(_QWORD *)(v4 + v5);
        }
        ++v8;
      }
      while ( v6 != (_DWORD)v8 );
      v5 = 0x1FFFFFFFE0LL;
    }
    return *(_QWORD *)(v4 + v5);
  }
  return result;
}
