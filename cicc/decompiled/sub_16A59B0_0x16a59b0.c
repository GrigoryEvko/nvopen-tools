// Function: sub_16A59B0
// Address: 0x16a59b0
//
__int64 __fastcall sub_16A59B0(__int64 *a1, __int64 *a2)
{
  unsigned __int64 v2; // rcx
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rax

  v2 = ((unsigned __int64)*((unsigned int *)a1 + 2) + 63) >> 6;
  if ( !v2 )
    return 0;
  v3 = *a1;
  v4 = *a2;
  v5 = 0;
  while ( (*(_QWORD *)(v4 + 8 * v5) & *(_QWORD *)(v3 + 8 * v5)) == 0 )
  {
    if ( v2 == ++v5 )
      return 0;
  }
  return 1;
}
