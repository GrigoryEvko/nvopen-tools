// Function: sub_85E7F0
// Address: 0x85e7f0
//
void __fastcall sub_85E7F0(__int64 *a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx

  for ( ; a1; a1 = (__int64 *)*a1 )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)a1 + 8) != 6 || (v4 = a1[2], v5 = *(_QWORD *)(v4 + 168), (*(_BYTE *)(v5 + 109) & 0x20) == 0) )
        sub_721090();
      if ( a2 )
        *(_QWORD *)(v5 + 240) = *(_QWORD *)(a2 + 88);
      else
        *(_BYTE *)(v5 + 110) &= 0xE7u;
      if ( a3 )
        break;
      a1 = (__int64 *)*a1;
      if ( !a1 )
        return;
    }
    while ( *(_BYTE *)(v4 + 140) == 12 )
      v4 = *(_QWORD *)(v4 + 160);
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v4 + 96LL) + 181LL) |= 8u;
  }
}
