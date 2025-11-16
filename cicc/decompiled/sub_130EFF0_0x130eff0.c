// Function: sub_130EFF0
// Address: 0x130eff0
//
int __fastcall sub_130EFF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rsi
  int result; // eax

  if ( *(_QWORD *)(a2 + 64) )
  {
    v3 = 0;
    do
    {
      v4 = 9 * v3++;
      result = sub_130B050(a1, *(_QWORD *)(a2 + 104) + 16 * v4);
    }
    while ( *(_QWORD *)(a2 + 64) > v3 );
  }
  return result;
}
