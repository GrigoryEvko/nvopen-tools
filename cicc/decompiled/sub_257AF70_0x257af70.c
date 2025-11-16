// Function: sub_257AF70
// Address: 0x257af70
//
__int64 __fastcall sub_257AF70(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned int v5; // r15d
  __int64 v6; // rdx

  v2 = *(unsigned int *)(a1 + 112);
  if ( (_DWORD)v2 )
  {
    v3 = 8 * v2;
    v4 = 0;
    v5 = 1;
    do
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v4);
        if ( !*(_DWORD *)(v6 + 24) )
          break;
        v4 += 8;
        if ( v3 == v4 )
          return v5;
      }
      if ( (unsigned __int8)sub_257AC00(a1, a2, (__int64 *)v6, 0) )
        v5 = 0;
      v4 += 8;
    }
    while ( v3 != v4 );
  }
  else
  {
    return 1;
  }
  return v5;
}
