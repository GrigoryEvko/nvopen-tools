// Function: sub_2D22A70
// Address: 0x2d22a70
//
__int64 __fastcall sub_2D22A70(__int64 a1, int a2, int a3)
{
  __int64 result; // rax
  __int64 i; // rdx
  __int64 v6; // rsi
  __int64 v7; // rdx

  result = *(_QWORD *)(a1 + 8);
  for ( i = 16LL * (unsigned int)(a2 - 1); i; i -= 16 )
  {
    *(_DWORD *)(*(_QWORD *)(i + result) + 4LL * *(unsigned int *)(i + result + 12) + 128) = a3;
    result = *(_QWORD *)(a1 + 8);
    v6 = result + i;
    if ( *(_DWORD *)(v6 + 12) != *(_DWORD *)(v6 + 8) - 1 )
      return result;
  }
  v7 = *(_QWORD *)result;
  result = *(unsigned int *)(result + 12);
  *(_DWORD *)(v7 + 4 * result + 120) = a3;
  return result;
}
