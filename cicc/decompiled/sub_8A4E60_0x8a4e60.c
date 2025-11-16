// Function: sub_8A4E60
// Address: 0x8a4e60
//
__int64 **__fastcall sub_8A4E60(__int64 **a1, _QWORD *a2, __int64 *a3, unsigned int a4, int *a5, __m128i *a6)
{
  int v6; // ebx
  __int64 v10; // r15
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // ecx

  v6 = a2[2] - 1;
  if ( v6 >= 0 )
  {
    v10 = 24LL * v6;
    do
    {
      if ( *a5 )
        break;
      v11 = a4 | 0x4000;
      v12 = v10 + *a2;
      if ( !v6 )
        v11 = a4;
      v13 = v11;
      if ( (*(_BYTE *)(v12 + 16) & 4) != 0 )
      {
        BYTE1(v13) = BYTE1(v11) | 0x20;
        v11 = v13;
      }
      if ( (*(_BYTE *)(v12 + 16) & 8) != 0 )
        v11 = (unsigned int)v11 | 0x80000;
      --v6;
      v10 -= 24;
      a1 = sub_8A2270((__int64)a1, *(__m128i **)(v12 + 8), *(_QWORD *)v12, a3, v11, a5, a6);
    }
    while ( v6 != -1 );
  }
  return a1;
}
