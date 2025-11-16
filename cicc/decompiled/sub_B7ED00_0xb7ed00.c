// Function: sub_B7ED00
// Address: 0xb7ed00
//
__int64 __fastcall sub_B7ED00(__int64 a1)
{
  int v1; // ebx
  unsigned int v2; // r15d
  __int64 v3; // r12
  __int64 (*v4)(); // rax

  v1 = *(_DWORD *)(a1 + 200) - 1;
  if ( v1 < 0 )
  {
    return 0;
  }
  else
  {
    v2 = 0;
    v3 = 8LL * v1;
    do
    {
      while ( 1 )
      {
        v4 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 192) + v3) + 32LL);
        if ( v4 != sub_97DD10 )
          break;
        --v1;
        v3 -= 8;
        if ( v1 == -1 )
          return v2;
      }
      --v1;
      v3 -= 8;
      v2 |= v4();
    }
    while ( v1 != -1 );
  }
  return v2;
}
