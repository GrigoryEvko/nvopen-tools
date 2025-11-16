// Function: sub_3981F30
// Address: 0x3981f30
//
__int64 __fastcall sub_3981F30(__int64 a1, __int64 a2, __int16 a3)
{
  __int64 *v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 result; // rax
  int v7; // edx
  __int16 v8; // cx

  v3 = *(__int64 **)(a2 + 8);
  if ( v3 )
  {
    v4 = *v3;
    do
    {
      v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v5 )
        break;
      if ( a3 == *(_WORD *)(v5 + 12) )
      {
        *(_WORD *)(a1 + 4) = a3;
        v7 = *(_DWORD *)(v5 + 8);
        v8 = *(_WORD *)(v5 + 14);
        *(_DWORD *)a1 = v7;
        *(_WORD *)(a1 + 6) = v8;
        switch ( v7 )
        {
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
          case 6:
          case 7:
          case 8:
          case 9:
          case 10:
            *(_QWORD *)(a1 + 8) = *(_QWORD *)(v5 + 16);
            result = a1;
            break;
          default:
            return a1;
        }
        return result;
      }
      v4 = *(_QWORD *)v5;
    }
    while ( (v4 & 4) == 0 );
  }
  *(_OWORD *)a1 = 0;
  return a1;
}
