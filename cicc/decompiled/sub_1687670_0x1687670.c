// Function: sub_1687670
// Address: 0x1687670
//
_DWORD *__fastcall sub_1687670(_DWORD *a1)
{
  _DWORD *result; // rax
  __int64 v2; // rsi
  __int64 v3; // rdi
  unsigned int v4; // edx
  unsigned int v5; // ecx
  int v6; // ecx
  unsigned int v7; // edx
  __int64 v8; // rcx
  int v9; // edi

  result = a1;
  if ( a1 )
  {
    v2 = *(_QWORD *)a1;
    v3 = (int)a1[2];
    if ( (unsigned int)v3 < *(_DWORD *)(*(_QWORD *)result + 80LL) )
    {
      v4 = result[3];
      if ( v4 )
      {
        _BitScanForward(&v5, v4);
        v6 = 1 << v5;
        result[3] = v6 ^ v4;
        if ( v4 == v6 )
        {
          v7 = v3 + 1;
          v8 = 4 * v3 + 4;
          do
          {
            result[2] = v7;
            if ( *(_DWORD *)(v2 + 80) <= v7 )
              break;
            ++v7;
            v9 = *(_DWORD *)(*(_QWORD *)(v2 + 96) + v8);
            v8 += 4;
            result[3] = v9;
          }
          while ( !v9 );
        }
      }
    }
  }
  return result;
}
