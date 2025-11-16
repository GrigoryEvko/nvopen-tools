// Function: sub_D46CA0
// Address: 0xd46ca0
//
__int64 __fastcall sub_D46CA0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r13
  int v4; // r12d
  unsigned int v5; // r15d
  __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 != a2 + 48 )
  {
    if ( !v2 )
      BUG();
    v3 = v2 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
      return 0;
    v4 = sub_B46E30(v3);
    if ( !v4 )
      return 0;
    v5 = 0;
    while ( 1 )
    {
      v6 = sub_B46EC0(v3, v5);
      if ( *(_BYTE *)(a1 + 84) )
      {
        v7 = *(_QWORD **)(a1 + 64);
        v8 = &v7[*(unsigned int *)(a1 + 76)];
        if ( v7 == v8 )
          return 1;
        while ( v6 != *v7 )
        {
          if ( v8 == ++v7 )
            return 1;
        }
      }
      else if ( !sub_C8CA60(a1 + 56, v6) )
      {
        return 1;
      }
      if ( v4 == ++v5 )
        return 0;
    }
  }
  return 0;
}
