// Function: sub_1455760
// Address: 0x1455760
//
__int64 __fastcall sub_1455760(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r13

  v3 = 1LL << a3;
  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 <= 0x40 )
  {
    *(_QWORD *)a1 = 0;
LABEL_3:
    *(_QWORD *)a1 |= v3;
    return a1;
  }
  sub_16A4EF0(a1, 0, 0);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    goto LABEL_3;
  *(_QWORD *)(*(_QWORD *)a1 + 8LL * (a3 >> 6)) |= v3;
  return a1;
}
