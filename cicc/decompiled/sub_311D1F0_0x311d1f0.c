// Function: sub_311D1F0
// Address: 0x311d1f0
//
__int64 __fastcall sub_311D1F0(__int64 a1)
{
  unsigned int v1; // edx
  unsigned int v2; // ecx
  unsigned __int64 v3; // rsi
  __int64 result; // rax
  int v5; // eax

  v1 = *(_DWORD *)a1;
  v2 = *(_DWORD *)(a1 + 4);
  v3 = *(_QWORD *)(a1 + 8);
  while ( 1 )
  {
    result = *(unsigned int *)(a1 - 16);
    if ( v1 >= (unsigned int)result
      && (v1 != (_DWORD)result || *(_DWORD *)(a1 - 12) <= v2)
      && (v1 > (unsigned int)result || *(_DWORD *)(a1 - 12) < v2 || v3 >= *(_QWORD *)(a1 - 8)) )
    {
      break;
    }
    *(_DWORD *)a1 = result;
    v5 = *(_DWORD *)(a1 - 12);
    a1 -= 16;
    *(_DWORD *)(a1 + 20) = v5;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a1 + 8);
  }
  *(_DWORD *)a1 = v1;
  *(_DWORD *)(a1 + 4) = v2;
  *(_QWORD *)(a1 + 8) = v3;
  return result;
}
