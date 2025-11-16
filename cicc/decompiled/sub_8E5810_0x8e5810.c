// Function: sub_8E5810
// Address: 0x8e5810
//
unsigned __int8 *__fastcall sub_8E5810(unsigned __int8 *a1, __int64 *a2, __int64 a3)
{
  unsigned __int8 v3; // al
  unsigned __int8 *v4; // r8
  int v6; // edi
  __int64 v7; // rdx
  int v9; // ecx

  v3 = *a1;
  v4 = a1;
  v6 = 0;
  if ( v3 == 110 )
  {
    v3 = v4[1];
    v6 = 1;
    ++v4;
  }
  v7 = 0;
  if ( (unsigned int)v3 - 48 <= 9 )
  {
    do
    {
      v9 = *++v4;
      v7 = (char)v3 - 48 + 10 * v7;
      v3 = v9;
    }
    while ( (unsigned int)(v9 - 48) <= 9 );
  }
  else if ( !*(_DWORD *)(a3 + 24) )
  {
    ++*(_QWORD *)(a3 + 32);
    ++*(_QWORD *)(a3 + 48);
    *(_DWORD *)(a3 + 24) = 1;
  }
  if ( v6 )
    v7 = -v7;
  *a2 = v7;
  return v4;
}
