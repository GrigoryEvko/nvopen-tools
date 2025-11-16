// Function: sub_E31BC0
// Address: 0xe31bc0
//
unsigned __int64 __fastcall sub_E31BC0(__int64 a1)
{
  unsigned __int64 v1; // rcx
  unsigned __int64 v2; // rax
  char v3; // al
  __int64 v4; // rsi

  v1 = 0;
  if ( !*(_BYTE *)(a1 + 49) )
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(a1 + 40);
      if ( v2 >= *(_QWORD *)(a1 + 24) )
        break;
      *(_QWORD *)(a1 + 40) = v2 + 1;
      v3 = *(_BYTE *)(*(_QWORD *)(a1 + 32) + v2);
      if ( v3 == 95 )
      {
        if ( v1 == -1 )
          break;
        return v1 + 1;
      }
      if ( (unsigned __int8)(v3 - 48) > 9u )
      {
        if ( (unsigned __int8)(v3 - 97) > 0x19u )
        {
          if ( (unsigned __int8)(v3 - 65) > 0x19u )
            break;
          v4 = v3 - 29;
        }
        else
        {
          v4 = v3 - 87;
        }
      }
      else
      {
        v4 = v3 - 48;
      }
      if ( is_mul_ok(0x3Eu, v1) && 62 * v1 <= ~v4 )
      {
        v1 = 62 * v1 + v4;
        if ( !*(_BYTE *)(a1 + 49) )
          continue;
      }
      break;
    }
  }
  *(_BYTE *)(a1 + 49) = 1;
  return 0;
}
