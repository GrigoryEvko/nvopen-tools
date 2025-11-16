// Function: sub_1319210
// Address: 0x1319210
//
int __fastcall sub_1319210(__int64 a1, __int64 a2)
{
  int result; // eax
  unsigned int v3; // r13d
  __int64 v4; // rdx

  result = dword_4F96B60;
  if ( dword_4F96B60 )
  {
    v3 = 0;
    do
    {
      v4 = v3++;
      result = sub_131C7E0(a1, a2 + 224 * v4 + 78984);
    }
    while ( dword_4F96B60 > v3 );
  }
  return result;
}
