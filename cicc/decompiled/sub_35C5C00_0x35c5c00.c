// Function: sub_35C5C00
// Address: 0x35c5c00
//
__int64 __fastcall sub_35C5C00(__int64 a1)
{
  unsigned __int64 v1; // rbx
  __int64 v2; // rax
  __int64 result; // rax
  __int64 i; // rdx
  unsigned __int64 v5; // rax

  v1 = **(_QWORD **)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)v1;
  if ( (*(_QWORD *)v1 & 4) == 0 && (*(_BYTE *)(v1 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v5 = v2 & 0xFFFFFFFFFFFFFFF8LL;
      v1 = v5;
      if ( (*(_BYTE *)(v5 + 44) & 4) == 0 )
        break;
      v2 = *(_QWORD *)v5;
    }
  }
  *(_QWORD *)(a1 + 32) = v1;
  sub_2E21F40((__int64 *)(a1 + 88), v1);
  result = *(_QWORD *)(a1 + 40);
  for ( i = result + 16LL * *(unsigned int *)(a1 + 48); result != i; *(_QWORD *)(result - 8) = 0 )
  {
    while ( *(_QWORD *)(result + 8) != v1 )
    {
      result += 16;
      if ( result == i )
        return result;
    }
    *(_DWORD *)(result + 4) = 0;
    result += 16;
  }
  return result;
}
