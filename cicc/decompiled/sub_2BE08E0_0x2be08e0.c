// Function: sub_2BE08E0
// Address: 0x2be08e0
//
__int64 __fastcall sub_2BE08E0(__int64 a1, int a2)
{
  __int64 result; // rax
  unsigned __int64 v4; // r12
  char v5; // di

  if ( !*(_QWORD *)(a1 + 280) )
    return 0;
  result = 0;
  v4 = 0;
  do
  {
    v5 = *(_BYTE *)(*(_QWORD *)(a1 + 272) + v4++);
    result = a2 * result + (int)sub_2BDCD80(v5, a2);
  }
  while ( *(_QWORD *)(a1 + 280) > v4 );
  return result;
}
