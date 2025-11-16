// Function: sub_5E9440
// Address: 0x5e9440
//
void __fastcall sub_5E9440(__int64 a1)
{
  __int64 *v1; // r12
  __int64 *v3; // rbx
  unsigned __int8 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned __int8 v7; // al

  v1 = *(__int64 **)(a1 + 112);
  if ( !v1 )
  {
    MEMORY[0x18] &= ~2u;
    BUG();
  }
  v3 = *(__int64 **)(a1 + 112);
  while ( (v3[3] & 2) == 0 )
  {
    v3 = (__int64 *)*v3;
    if ( !v3 )
    {
      v4 = 3;
      do
      {
        v5 = v1[1];
        v6 = *(_QWORD *)(v5 + 16);
        if ( (*(_BYTE *)(v6 + 96) & 2) != 0 && a1 != v6 )
        {
          sub_5E9440();
          v5 = v1[1];
        }
        v7 = sub_87D630(0, v5, v1);
        if ( *(__int64 **)(a1 + 112) == v1 || v7 < v4 )
        {
          v4 = v7;
          v3 = v1;
        }
        else if ( v7 == v4 && (v3[3] & 1) == 0 )
        {
          if ( (v1[3] & 1) != 0 )
          {
            v3 = v1;
          }
          else if ( (*(_BYTE *)(*(_QWORD *)(v1[1] + 16) + 96LL) & 2) == 0
                 && (*(_BYTE *)(*(_QWORD *)(v3[1] + 16) + 96LL) & 2) != 0 )
          {
            v3 = v1;
          }
        }
        v1 = (__int64 *)*v1;
      }
      while ( v1 );
      *((_BYTE *)v3 + 24) |= 2u;
      return;
    }
  }
}
