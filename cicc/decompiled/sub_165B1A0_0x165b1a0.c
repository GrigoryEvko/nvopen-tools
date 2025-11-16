// Function: sub_165B1A0
// Address: 0x165b1a0
//
__int64 __fastcall sub_165B1A0(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  __int64 *v2; // rcx
  __int64 *v3; // rax
  __int64 result; // rax

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = (__int64 *)(v1 - 24);
  v3 = (__int64 *)(v1 - 72);
  if ( (*a1 & 4) != 0 )
    v3 = v2;
  result = *v3;
  if ( *(_BYTE *)(result + 16) )
    return 0;
  return result;
}
