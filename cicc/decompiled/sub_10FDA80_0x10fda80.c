// Function: sub_10FDA80
// Address: 0x10fda80
//
__int64 __fastcall sub_10FDA80(__int64 a1, unsigned int a2, unsigned int a3)
{
  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    a2 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = 0;
  }
  if ( a3 == a2 )
    return a1;
  if ( a3 <= 0x3F && a2 <= 0x40 )
  {
    *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)a3 + 64 - (unsigned __int8)a2) << a3;
    return a1;
  }
  sub_C43C90((_QWORD *)a1, a3, a2);
  return a1;
}
