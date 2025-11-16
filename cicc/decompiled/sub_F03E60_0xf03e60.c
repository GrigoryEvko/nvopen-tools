// Function: sub_F03E60
// Address: 0xf03e60
//
__int64 __fastcall sub_F03E60(
        unsigned int a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int8 a7)
{
  __int64 result; // rax
  __int64 v9; // r14
  unsigned int v10; // r11d
  __int64 v11; // rcx
  __int64 v12; // r13
  unsigned int v13; // eax
  unsigned int v14; // edx
  unsigned int v15; // edi
  unsigned int v16; // ebx
  unsigned int v17; // esi

  result = 0;
  if ( a1 )
  {
    v9 = a1;
    v10 = a1;
    v11 = 0;
    v12 = 0;
    v14 = (a2 + (unsigned int)a7) % a1;
    v13 = (a2 + (unsigned int)a7) / a1;
    v15 = 0;
    do
    {
      while ( 1 )
      {
        v16 = v15;
        v17 = v13 + (v14 > (unsigned int)v11);
        *(_DWORD *)(a5 + 4 * v11) = v17;
        v15 += v17;
        if ( a1 == v10 && v15 > a6 )
          break;
        if ( v9 == ++v11 )
          goto LABEL_7;
      }
      v10 = v11++;
      v12 = a6 - v16;
    }
    while ( v9 != v11 );
LABEL_7:
    if ( a7 )
      --*(_DWORD *)(a5 + 4LL * v10);
    return (v12 << 32) | v10;
  }
  return result;
}
