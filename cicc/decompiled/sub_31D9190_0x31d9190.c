// Function: sub_31D9190
// Address: 0x31d9190
//
__int64 __fastcall sub_31D9190(__int64 *a1)
{
  unsigned int v1; // eax
  __int64 v2; // rdx
  __int64 v3; // r8

  v1 = *((_DWORD *)a1 + 2);
  v2 = *a1;
  if ( v1 > 0x40 )
    return *(_QWORD *)v2;
  v3 = 0;
  if ( v1 )
    return v2 << (64 - (unsigned __int8)v1) >> (64 - (unsigned __int8)v1);
  return v3;
}
