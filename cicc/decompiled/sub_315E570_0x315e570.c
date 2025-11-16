// Function: sub_315E570
// Address: 0x315e570
//
__int64 __fastcall sub_315E570(__int64 a1, int a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // r13
  int v5; // ebx
  unsigned int v6; // r14d
  __int64 v7; // rax

  result = 0;
  if ( a2 )
  {
    result = sub_24F3200(a1);
    if ( !(_BYTE)result )
    {
      v3 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v3 == a1 + 48 )
        return 1;
      if ( !v3 )
        BUG();
      v4 = v3 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
        return 1;
      v5 = sub_B46E30(v4);
      if ( !v5 )
      {
        return 1;
      }
      else
      {
        v6 = 0;
        while ( 1 )
        {
          v7 = sub_B46EC0(v4, v6);
          result = sub_315E570(v7, (unsigned int)(a2 - 1));
          if ( !(_BYTE)result )
            break;
          if ( v5 == ++v6 )
            return 1;
        }
      }
    }
  }
  return result;
}
