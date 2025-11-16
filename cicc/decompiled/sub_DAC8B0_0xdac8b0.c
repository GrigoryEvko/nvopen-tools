// Function: sub_DAC8B0
// Address: 0xdac8b0
//
__int64 __fastcall sub_DAC8B0(__int64 a1, _QWORD *a2)
{
  unsigned __int64 v2; // r8

  do
  {
    v2 = (unsigned __int64)a2;
    a2 = (_QWORD *)*a2;
  }
  while ( a2 );
  return sub_DAC210(a1, v2);
}
