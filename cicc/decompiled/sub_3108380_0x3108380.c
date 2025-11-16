// Function: sub_3108380
// Address: 0x3108380
//
__int64 __fastcall sub_3108380(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  unsigned __int8 *v5; // rdi
  unsigned int v6; // eax

  v3 = 3;
  if ( byte_5031DC8[0] )
  {
    if ( *(_BYTE *)a2 == 85 )
    {
      v5 = *(unsigned __int8 **)(a2 - 32);
      if ( v5 )
      {
        v3 = *v5;
        if ( (_BYTE)v3 )
          return 3;
        if ( *((_QWORD *)v5 + 3) != *(_QWORD *)(a2 + 80) )
          return 3;
        v6 = sub_3108960(v5, a2, a3);
        if ( v6 > 0xB )
        {
          return 3;
        }
        else if ( ((1LL << v6) & 0xEE3) == 0 )
        {
          return 3;
        }
      }
    }
  }
  return v3;
}
