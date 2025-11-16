// Function: sub_97E0F0
// Address: 0x97e0f0
//
__int64 __fastcall sub_97E0F0(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx

  LOBYTE(a5) = (*(_BYTE *)(*(_QWORD *)a2 + 8LL) & 0xFD) != 12 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 7;
  if ( (_BYTE)a5 )
    return 0;
  v5 = a2 + 8;
  v6 = a2 + 8LL * a1;
  if ( a2 + 8 == v6 )
  {
    return 1;
  }
  else
  {
    while ( (*(_BYTE *)(*(_QWORD *)v5 + 8LL) & 0xFD) == 0xC )
    {
      v5 += 8;
      if ( v6 == v5 )
        return 1;
    }
  }
  return a5;
}
