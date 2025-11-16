// Function: sub_1D16990
// Address: 0x1d16990
//
__int64 __fastcall sub_1D16990(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rcx

  v1 = *(unsigned int *)(a1 + 56);
  result = 0;
  if ( (_DWORD)v1 )
  {
    v3 = *(_QWORD *)(a1 + 32);
    v4 = v3 + 40 * v1;
    do
    {
      if ( *(_WORD *)(*(_QWORD *)v3 + 24LL) != 48 )
        return 0;
      v3 += 40;
    }
    while ( v4 != v3 );
    return 1;
  }
  return result;
}
