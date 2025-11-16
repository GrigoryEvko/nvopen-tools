// Function: sub_A73E30
// Address: 0xa73e30
//
__int64 __fastcall sub_A73E30(__int64 a1, int a2, _DWORD *a3)
{
  int v3; // eax
  unsigned int v4; // r12d
  int v5; // ebx
  _QWORD *v6; // r14
  int v7; // r15d
  unsigned int v8; // eax

  v3 = a2 + 7;
  if ( a2 >= 0 )
    v3 = a2;
  v4 = ((int)*(unsigned __int8 *)(a1 + (v3 >> 3) + 28) >> (a2 % 8)) & 1;
  if ( v4 )
  {
    if ( a3 )
    {
      v5 = *(_DWORD *)(a1 + 8);
      if ( v5 )
      {
        v6 = (_QWORD *)(a1 + 48);
        v7 = 0;
        while ( 1 )
        {
          v8 = sub_A73170(v6, a2);
          if ( (_BYTE)v8 )
            break;
          ++v7;
          ++v6;
          if ( v5 == v7 )
            return v4;
        }
        v4 = v8;
        *a3 = v7 - 1;
      }
    }
  }
  return v4;
}
