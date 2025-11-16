// Function: sub_13D0120
// Address: 0x13d0120
//
__int64 __fastcall sub_13D0120(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
    sub_16A4EF0(a1, 0, 0);
  else
    *(_QWORD *)a1 = 0;
  if ( !a3 )
    return a1;
  if ( a3 > 0x40 )
  {
    sub_16A5260(a1, 0, a3);
    return a1;
  }
  else
  {
    v4 = *(_QWORD *)a1;
    v5 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a3);
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    {
      *(_QWORD *)a1 = v4 | v5;
      return a1;
    }
    *(_QWORD *)v4 |= v5;
    return a1;
  }
}
