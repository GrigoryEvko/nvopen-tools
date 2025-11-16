// Function: sub_31C5440
// Address: 0x31c5440
//
__int64 __fastcall sub_31C5440(unsigned int a1)
{
  __int64 result; // rax
  int v2; // edx
  int v3; // ecx
  unsigned int v4; // eax
  unsigned int v5; // esi

  result = 0;
  if ( a1 )
  {
    v2 = 5;
    v3 = 1;
    v4 = a1 >> 1;
    do
    {
      v5 = v4 >> v3;
      v3 *= 2;
      v4 |= v5;
      --v2;
    }
    while ( v2 );
    return a1 & ~v4;
  }
  return result;
}
