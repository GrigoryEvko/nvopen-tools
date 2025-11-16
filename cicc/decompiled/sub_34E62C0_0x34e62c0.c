// Function: sub_34E62C0
// Address: 0x34e62c0
//
__int64 __fastcall sub_34E62C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 i; // r8

  v2 = *(_QWORD *)(a2 + 64);
  result = *(unsigned int *)(a2 + 72);
  for ( i = v2 + 8 * result; i != v2; v2 += 8 )
  {
    result = *(_QWORD *)(a1 + 200) + 392LL * *(int *)(*(_QWORD *)v2 + 24LL);
    if ( (*(_BYTE *)result & 1) == 0 && *(_QWORD *)(result + 16) != a2 )
      *(_BYTE *)result &= 0xF3u;
  }
  return result;
}
