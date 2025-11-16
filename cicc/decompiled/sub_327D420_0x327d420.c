// Function: sub_327D420
// Address: 0x327d420
//
__int64 __fastcall sub_327D420(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
    sub_C43690(a1, 0, 0);
  else
    *(_QWORD *)a1 = 0;
  if ( a3 == a4 )
    return a1;
  if ( a3 > 0x3F || a4 > 0x40 )
  {
    sub_C43C90((_QWORD *)a1, a3, a4);
    return a1;
  }
  else
  {
    v6 = *(_QWORD *)a1;
    v7 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)a3 - (unsigned __int8)a4 + 64) << a3;
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    {
      *(_QWORD *)a1 = v6 | v7;
      return a1;
    }
    *(_QWORD *)v6 |= v7;
    return a1;
  }
}
