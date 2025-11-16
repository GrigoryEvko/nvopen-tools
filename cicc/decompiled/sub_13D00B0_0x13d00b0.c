// Function: sub_13D00B0
// Address: 0x13d00b0
//
__int64 __fastcall sub_13D00B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // r13

  v2 = 1LL << ((unsigned __int8)a2 - 1);
  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 <= 0x40 )
  {
    *(_QWORD *)a1 = 0;
LABEL_3:
    *(_QWORD *)a1 |= v2;
    return a1;
  }
  sub_16A4EF0(a1, 0, 0);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    goto LABEL_3;
  *(_QWORD *)(*(_QWORD *)a1 + 8LL * ((a2 - 1) >> 6)) |= v2;
  return a1;
}
