// Function: sub_161F650
// Address: 0x161f650
//
__int64 __fastcall sub_161F650(__int64 *a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *a1;
  v3 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( v3 == *a1 )
    return 0;
  while ( *(_DWORD *)v2 != a2 )
  {
    v2 += 16;
    if ( v3 == v2 )
      return 0;
  }
  return *(_QWORD *)(v2 + 8);
}
